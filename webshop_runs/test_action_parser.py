from action_parser import parse_webshop_action


def test_valid_commands_pass_through():
    assert parse_webshop_action("search[usb microphone]") == "search[usb microphone]"
    assert parse_webshop_action("click[Buy Now]") == "click[Buy Now]"
    assert parse_webshop_action("think[compare prices]") == "think[compare prices]"


def test_action_prefixes_are_removed():
    assert parse_webshop_action(
        "Action: search[hair extension]"
    ) == "search[hair extension]"
    assert parse_webshop_action(
        "Action: Action: search[60 capsules]"
    ) == "search[60 capsules]"
    assert parse_webshop_action(
        "Reasoning first\nAction: click[B078ABC]"
    ) == "click[B078ABC]"


def test_prose_is_not_fabricated_into_an_action():
    prose = "I think you should search for a microphone."
    assert parse_webshop_action(prose) == prose
    assert parse_webshop_action("") == ""


if __name__ == "__main__":
    test_valid_commands_pass_through()
    test_action_prefixes_are_removed()
    test_prose_is_not_fabricated_into_an_action()
    print("3 tests passed")
