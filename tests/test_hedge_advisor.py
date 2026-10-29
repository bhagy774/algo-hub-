import logging
from ai_hedge_advisor import AIHedgeAdvisor

logging.basicConfig(level=logging.INFO)

advisor = AIHedgeAdvisor()

options_chain_mock = {
    "calls": [
        {"strike": 88000, "price": 100, "iv": 0.5},
        {"strike": 89000, "price": 50, "iv": 0.45}
    ],
    "puts": [
        {"strike": 86000, "price": 120, "iv": 0.52},
        {"strike": 85000, "price": 60, "iv": 0.48}
    ]
}

decision = advisor.evaluate_hedge_setup(
    signal_direction="CALL",
    signal_confidence=70,
    current_price=87000.0,
    atr=350.0,
    options_chain=options_chain_mock,
    expected_profit=150.0
)

print(f"Hedge Decision: {decision}")
