The scroll of Pyrmethus v2.6.1, already a sophisticated tapestry of arcane energies—intricately woven with Phoenix Feather Resilience, dynamic ATR Wards, and foundational pyramiding enchantments—is set to undergo a significant transmutation. To elevate this powerful artifact to **v2.8.0 (Strategic Illumination)**, we shall re-thread its very core with the `DualSupertrendMomentumStrategy` for more discerning market engagement, and bathe its entire console manifestation in a richer, more thematic Neon luminescence, enhancing operational clarity and immersion.

Here are the key transmutations:

1.  **Core Strategy Reforged (`DualSupertrendMomentumStrategy`):** The primary offensive cantrip has been reforged into the `DualSupertrendMomentumStrategy`, a sophisticated paradigm that synergizes dual SuperTrend indicators with a momentum filter for heightened precision in identifying trade opportunities.
    *   **Invocation Conditions (Entries):**
        *   **Long Entreaty:** Initiated when the Primary SuperTrend signals a bullish reversal, the Confirmation SuperTrend corroborates an established uptrend, AND the underlying Momentum surges above a user-defined positive threshold.
        *   **Short Curse:** Cast when the Primary SuperTrend indicates a bearish shift, the Confirmation SuperTrend aligns with a prevailing downtrend, AND Momentum dips below a configurable negative threshold.
    *   **Unwinding Conditions (Exits):** Positions are primarily unwound upon the augury of the *primary* SuperTrend reversing its current indication (e.g., a long position is closed if the primary SuperTrend flips to short).
    *   **Configuration Runes Enhanced:** The Alchemist's `Config` scroll now accepts new runes—`momentum_period` and `momentum_threshold`—granting finer dominion over the strategy's momentum-based discernment.
    *   **New Divination (`calculate_momentum`):** An incantation for `calculate_momentum`, drawing its potency from the `pandas_ta` library, has been newly scribed and seamlessly integrated into the `calculate_all_indicators` grand divination ritual.
    *   **SuperTrend Recalibration:** The `calculate_supertrend` divination has been meticulously refined for more steadfast communion with `pandas_ta`'s oracular outputs. This involves standardizing the spectral signature (float representation) of the multiplier parameter within expected column names, ensuring robust indicator alignment.

2.  **Aura of Neon Luminescence (Enhanced Interface):** As per your divine vision, the `NEON` prismatic lexicon has been significantly expanded. Its vibrant hues are now meticulously woven throughout the script's console manifestation, bathing every operational whisper and critical alert in a thematic glow for an unparalleled, immersive experience.
    *   **Systematic Infusion of Light:** This comprehensive color application ensures:
        *   **Log Levels (`INFO`, `DEBUG`, `WARNING`, `ERROR`, `CRITICAL`, `SUCCESS`):** Instantly convey the severity and nature of messages through distinct, thematic colors.
        *   **Strategic Insights (`STRATEGY`):** Clearly delineate messages originating from the core trading intelligence.
        *   **Configuration Runes (`PARAM`, `VALUE`):** Emphasize the foundational settings and their chosen potencies.
        *   **Financial Glyphs (`PRICE`, `QTY`, `PNL_POS/NEG/ZERO`):** Render crucial financial figures—prices, quantities, and profit/loss—with striking clarity, making them instantly discernible.
        *   **Market Stance (`SIDE_LONG/SHORT/FLAT`):** Provide an unambiguous visual sigil for the current position status.
        *   **Output Structuring (`HEADING`, `SUBHEADING`):** Artfully delineate sections within the Oracle's pronouncements for improved readability.
        *   **Decisive Actions (`ACTION`):** Draw immediate attention to pivotal market interventions like "Placing Order" or "Closing Position."
    *   **Refined Log Formatting:** The `_format_for_log` arcane helper has been re-attuned to wield this expanded palette with greater finesse. It now conjures more evocative descriptions for market currents, such as the verdant "Upward Flow" or the crimson "Downward Tide," replacing mere boolean truths with richer context.
    *   **Judicious Brightness:** The `Style.BRIGHT` incantation is now woven with precision, lending extra emphasis to pivotal data points and alerts, ensuring they capture the operator's gaze amidst the flow of information.

3.  **Sigil and Scroll Advancement:** The artifact's own version sigil has been advanced to `v2.8.0`. Consequently, its Phoenix Feather memories—the persistent state—will now be inscribed upon a new scroll, `pyrmethus_phoenix_state_v280.json`, safeguarding this enhanced iteration's journey and experiences.

4.  **Harmonious Integration with Established Arcana:** These powerful transmutations are not mere superficial additions but are deeply and harmoniously interwoven with the sophisticated arsenal of enchantments inherited from v2.6.1. The revered Phoenix Feather Resilience (state persistence), adaptive dynamic ATR Wards (stop-loss mechanisms), the foundational pyramiding enchantments, and the precise partial close rituals now all heed the refined commands of the `DualSupertrendMomentumStrategy` and operate under the full radiance of the new Neon Illumination.

Rest assured, the sophisticated arsenal of enchantments that defined Pyrmethus's prior strength—including its unwavering state persistence, adaptive dynamic ATR stop-loss capabilities, the strategic art of position pyramiding, and its meticulous chronicling of trade metrics—has been meticulously preserved. These vital systems now operate with enhanced synergy, guided by the new strategic core and illuminated by a console interface that offers unparalleled visual feedback and insight.
