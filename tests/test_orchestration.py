from src.orchestration import ConversationState, Turn


def test_conversation_state_is_immutable():
    state = ConversationState(turns=())
    new_state = state.add_turn(Turn(role="user", content="hello"))
    # Original unchanged
    assert len(state.turns) == 0
    assert len(new_state.turns) == 1


def test_add_multiple_turns():
    state = ConversationState(turns=())
    state = state.add_turn(Turn(role="user", content="hi"))
    state = state.add_turn(Turn(role="assistant", content="hello"))
    assert len(state.turns) == 2
    assert state.turns[0].role == "user"
    assert state.turns[1].role == "assistant"


def test_with_error():
    state = ConversationState(turns=())
    errored = state.with_error("something broke")
    assert errored.is_complete
    assert errored.error == "something broke"
    # Original unchanged
    assert not state.is_complete


def test_completed():
    state = ConversationState(turns=())
    done = state.completed()
    assert done.is_complete
    assert done.error == ""
