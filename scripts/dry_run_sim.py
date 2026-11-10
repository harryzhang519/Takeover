import sys
import os

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.devig import devig_shin_n
from src.core.risk import calculate_baker_mchale_shrinkage

def run_dry_run():
    print("=== BREEZE / TAKEOVER: TRADING FLOOR DRY RUN ===")
    
    # Mock incoming market odds from a sportsbook / exchange feed (e.g., -110 American odds = ~1.909 decimal)
    # Let's test a two-way market with standard vig
    raw_decimal_odds = [1.91, 1.91]
    print(1, f"1. Raw Ingested Odds (Decimal): {raw_decimal_odds}")
    
    # 2. De-vig market using Shin's model (returns fair probs and market friction parameter z)
    fair_probs, z = devig_shin_n(raw_decimal_odds)
    print(2, f"2. De-Vigged Fair Probabilities: {[round(p, 4) for p in fair_probs]}")
    print(f"   - Market Friction Parameter (z): {round(z, 4)}")
    
    # 3. Simulate model probability vs market fair probability (finding edge / +EV)
    # Suppose our quantitative model estimates true probability at 54% (0.54)
    model_prob = 0.54
    offered_odds = 1.91
    implied_prob = 1 / offered_odds
    edge = model_prob - implied_prob
    print(3, f"3. Edge Detection:")
    print(f"   - Model True Probability: {model_prob * 100}%")
    print(f"   - Market Implied Probability: {round(implied_prob * 100, 2)}%")
    print(f"   - Calculated Edge: +{round(edge * 100, 2)}%")
    
    # 4. Check Risk Constraints & Kelly Sizing
    # Desk Rules: Odds floor >= -130 (Decimal 1.769), straight singles only, max 2.0u exposure
    decimal_floor = 1.769
    if offered_odds < decimal_floor:
        print("❌ REJECTED: Odds violate the -130 floor constraint.")
        return

    # Calculate shrunken Kelly stake via Baker & McHale parameter uncertainty
    # b = net odds (decimal - 1), p = model_prob, sigma = estimation uncertainty
    b = offered_odds - 1.0
    recommended_stake = calculate_baker_mchale_shrinkage(b=b, p=model_prob, sigma=0.02)
    
    # Enforce quarter-Kelly and 2.0u desk cap
    capped_stake = min(recommended_stake * 0.25, 2.0)
    print(4, f"4. Risk Management & Sizing:")
    print(f"   - Recommended Shrunken Stake: {round(capped_stake, 2)}u (Capped at 2.0u max)")

    # 5. Simulate Captain HZ HITL Gate
    print(5, f"5. Captain HZ HITL Gate Intercept:")
    print(f"   [HITL PAUSE] Qualified +EV pick verified. Awaiting Captain HZ sign-off...")
    print(f"   -> [SIMULATED APPROVAL GRANTED]")
    print("=== DRY RUN COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_dry_run()
