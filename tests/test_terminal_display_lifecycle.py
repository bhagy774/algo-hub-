import unittest

try:
    from terminal_decision_display import format_decision_audit
except ImportError:
    from current_terminal_display import format_decision_audit


class TerminalLifecycleDisplayTests(unittest.TestCase):
    def render(self, outcome):
        return format_decision_audit(
            {"direction": "CALL", "execution_allowed": True}, "BTCUSDT",
            stage="ORDER_SUBMISSION", order_outcome=outcome,
            position_state={"live": [], "paper": []},
        )

    def test_executor_success_does_not_claim_submission_fill_or_protection(self):
        text = self.render({"success": True, "position": {"id": "private-id", "status": "OPEN"}})
        self.assertIn("success field: true", text)
        self.assertIn("no distinct submitted flag supplied", text)
        self.assertIn("Venue acknowledgement: UNKNOWN", text)
        self.assertIn("Fill confirmation: not supplied", text)
        self.assertIn("Protective-order status: UNKNOWN", text)
        self.assertIn("Close status: not supplied", text)
        self.assertNotIn("private-id", text)

    def test_explicit_lifecycle_fields_are_reported_without_identifiers(self):
        text = self.render({
            "success": True,
            "reason": "request for order_id=secret-order rejected; account number=hidden",
            "submitted": True,
            "acknowledged": "accepted",
            "fill_status": "filled",
            "close_status": "open",
            "protection": {"stop_loss": {"status": "working", "id": "secret-sl"},
                           "take_profit": {"status": "working", "id": "secret-tp"}},
            "position": {"id": "private-id", "status": "OPEN", "fill_price": 100.5,
                         "filled_quantity": 2},
        })
        self.assertIn("Reported submitted field: True", text)
        self.assertIn("Reported venue acknowledgement field: accepted", text)
        self.assertIn("fill_status field: filled", text)
        self.assertIn("close_status field: open", text)
        self.assertIn("stop-loss=", text)
        self.assertIn("take-profit=", text)
        self.assertIn("reported fill price: 100.5", text)
        self.assertIn("reported filled quantity: 2", text)
        for secret in ("private-id", "secret-sl", "secret-tp", "secret-order", "hidden"):
            self.assertNotIn(secret, text)


if __name__ == "__main__":
    unittest.main()
