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

from unittest.mock import MagicMock

from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.genai import types

from app.agent import escalate, polite_decline, root_agent, router, save_query


def test_save_query():
    mock_context = MagicMock(spec=Context)
    mock_context.state = {}
    generator = save_query._func(mock_context, "Track package 123")
    events = list(generator)
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, Event)
    assert event.output == "Track package 123"
    assert event.actions.state_delta["user_query"] == "Track package 123"


def test_save_query_repeats():
    mock_context = MagicMock(spec=Context)
    # Turn 1
    mock_context.state = {"queries_history": []}
    events = list(save_query._func(mock_context, "same question"))
    assert events[0].actions.state_delta["escalate_due_to_repeats"] is False

    # Turn 2
    mock_context.state = {"queries_history": ["same question"]}
    events = list(save_query._func(mock_context, "same question"))
    assert events[0].actions.state_delta["escalate_due_to_repeats"] is False

    # Turn 3
    mock_context.state = {"queries_history": ["same question", "same question"]}
    events = list(save_query._func(mock_context, "same question"))
    assert events[0].actions.state_delta["escalate_due_to_repeats"] is True


def test_router_shipping():
    mock_context = MagicMock(spec=Context)
    mock_context.state = {"inquiry_category": {"category": "shipping"}}
    event = router._func(mock_context)
    assert isinstance(event, Event)
    assert event.actions.route == "shipping"


def test_router_unrelated():
    mock_context = MagicMock(spec=Context)
    mock_context.state = {"inquiry_category": {"category": "unrelated"}}
    event = router._func(mock_context)
    assert isinstance(event, Event)
    assert event.actions.route == "unrelated"


def test_router_escalate_due_to_repeats():
    mock_context = MagicMock(spec=Context)
    mock_context.state = {
        "escalate_due_to_repeats": True,
        "inquiry_category": {"category": "shipping"},
    }
    event = router._func(mock_context)
    assert isinstance(event, Event)
    assert event.actions.route == "escalate"


def test_router_escalate_classifier():
    mock_context = MagicMock(spec=Context)
    mock_context.state = {
        "escalate_due_to_repeats": False,
        "inquiry_category": {"category": "escalate"},
    }
    event = router._func(mock_context)
    assert isinstance(event, Event)
    assert event.actions.route == "escalate"


def test_polite_decline():
    mock_context = MagicMock(spec=Context)
    event = polite_decline._func(mock_context)
    assert isinstance(event, Event)
    assert "shipping-related queries" in event.output
    assert isinstance(event.content, types.Content)


def test_escalate():
    mock_context = MagicMock(spec=Context)
    event = escalate._func(mock_context)
    assert isinstance(event, Event)
    assert "connecting you to a human agent" in event.output
    assert isinstance(event.content, types.Content)


def test_workflow_structure():
    assert root_agent.name == "customer_support_workflow"
    assert len(root_agent.edges) > 0
