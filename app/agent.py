# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Literal, Union, Any
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.workflow import START, Edge, Workflow, node
from google.genai import types


# Modify edges annotation to allow any tuple structure (e.g., 3-tuple routed edges)
Workflow.model_fields["edges"].annotation = list[Union[Edge, tuple[Any, ...]]]
Workflow.model_rebuild(force=True)


# Monkey-patch Edge to support chain classmethod
@classmethod
def _edge_chain(cls, *nodes):
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


Edge.chain = _edge_chain

# Monkey-patch ADK's graph engine to support 3-tuple (from, to, route) edges
import google.adk.workflow._graph as graph_module

original_process_chain = graph_module._process_chain


def custom_process_chain(chain, node_map, graph_edges):
    if (
        len(chain) == 3
        and not graph_module.is_node_like(chain[2])
        and isinstance(chain[2], (str, int, bool))
    ):
        from_node = graph_module._get_or_build_node(chain[0], node_map)
        to_node = graph_module._get_or_build_node(chain[1], node_map)
        route = chain[2]
        graph_edges.append(
            graph_module.Edge(
                from_node=from_node,
                to_node=to_node,
                route=route,
            )
        )
    else:
        original_process_chain(chain, node_map, graph_edges)


graph_module._process_chain = custom_process_chain


class InquiryCategory(BaseModel):
    category: Literal["shipping", "unrelated"] = Field(
        description="The category of the user inquiry. Set to 'shipping' if the inquiry is about shipping rates, package tracking, delivery status, or product returns. Set to 'unrelated' for all other inquiries (e.g. general chit-chat, unrelated questions)."
    )


@node
def save_query(node_input: str):
    yield Event(output=node_input, state={"user_query": node_input})  # type: ignore


classifier_agent = LlmAgent(
    name="classifier_agent",
    model="gemini-flash-latest",
    instruction=(
        "You are a customer query classifier for a shipping company.\n"
        "Analyze the user query and determine if it is related to shipping (such as rates, tracking, delivery status, or returns) or unrelated.\n"
        "Set `category` to 'shipping' if it is related to shipping, or 'unrelated' if it is unrelated."
    ),
    output_schema=InquiryCategory,
    output_key="inquiry_category",
)


@node
def router(ctx: Context) -> Event:
    inquiry_category = ctx.state.get("inquiry_category")
    category = inquiry_category.get("category") if inquiry_category else "unrelated"
    user_query = ctx.state.get("user_query")
    return Event(output=user_query, route=category)  # type: ignore


faq_agent_instruction = """You are a customer support agent for a shipping company.
Answer the user's shipping query politely and accurately based on the official FAQ knowledge base below:

1. Shipping Rates:
   - 🚚 Standard Shipping: Just $5.99! (Your package will arrive in 3-5 business days).
   - ⚡ Express Shipping: Only $14.99! (Super fast delivery in 1-2 business days).
   - 🎉 FREE SHIPPING: Spend over $50.00 and your shipping is absolutely free! 🥳

2. Package Tracking:
   - Customers can track their packages on our website using the tracking number provided in the confirmation email.
   - Updates are posted in real-time as the package is scanned at transit points.

3. Delivery:
   - Deliveries are made Monday through Saturday between 8:00 AM and 8:00 PM. No deliveries are made on Sundays or major holidays.
   - Signature is required for packages valued over $100.00.

4. Returns:
   - Items can be returned within 30 days of delivery.
   - Return shipping is free if using our pre-paid return label.
   - Refunds are processed within 5-7 business days after the return is received at our warehouse.

Answer shipping rates queries with extra enthusiasm and emojis, highlighting the free shipping threshold! Answer only using the information above. If the query cannot be answered by this information, answer politely that you don't know or ask for clarification."""

shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model="gemini-flash-latest",
    instruction=faq_agent_instruction,
)


@node
def polite_decline(ctx: Context) -> Event:
    text = (
        "I'm sorry, but I can only assist with shipping-related queries "
        "(such as rates, tracking, delivery, and returns). Let me know "
        "if you have any questions about these topics!"
    )
    return Event(
        content=types.Content(role="model", parts=[types.Part.from_text(text=text)]),
        output=text,
    )


root_agent = Workflow(
    name="customer_support_workflow",
    edges=[
        *Edge.chain(START, save_query, classifier_agent, router),
        (router, shipping_faq_agent, "shipping"),
        (router, polite_decline, "unrelated"),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
