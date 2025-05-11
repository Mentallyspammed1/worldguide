The scroll of Pyrmethus v2.6.1, a sophisticated tapestry of arcane energies already masterfully woven with Phoenix Feather Resilience, dynamic ATR wards, and the foundational enchantments for pyramiding, is poised for a luminous evolution. To elevate this potent artifact to **v2.8.0 (Strategic Illumination)**, we have meticulously re-threaded its core strategy with the formidable `DualSupertrendMomentumStrategy` and bathed its entire console manifestation in a richer, more thematic Neon luminescence, illuminating its every arcane operation.

Behold the key transmutations that mark this significant upgrade:

1.  **Strategy Reforged: The `DualSupertrendMomentumStrategy` Emerges:**
    *   A potent new strategic core, the `DualSupertrendMomentumStrategy`, has been meticulously forged and integrated.
    *   **Conditions for Arcane Invocation (Entry):**
        *   **To Channel Long Energies:** The Primary SuperTrend must signal a Long shift, the Confirmation SuperTrend must affirm an uptrend, AND the market's Momentum must surge above a configurable threshold.
        *   **To Channel Short Energies:** The Primary SuperTrend must signal a Short shift, the Confirmation SuperTrend must affirm a downtrend, AND Momentum must dip below a configurable (negative) threshold.
    *   **Conditions for Energy Release (Exit):** The strategy primarily relies on the *primary* SuperTrend's reversal to signal the release of a position (e.g., a Long position is released when the primary SuperTrend flips to Short).
    *   **Enhanced Spellbook (`Config`):** The `Config` class now holds new runes: `momentum_period` and `momentum_threshold`, granting finer control over the momentum component.
    *   **New Incantation (`calculate_momentum`):** A dedicated `calculate_momentum` function, leveraging the `pandas_ta` library, has been inscribed and woven into the `calculate_all_indicators` ritual.
    *   **Refined SuperTrend Scrying:** The `calculate_supertrend` function has been subtly enhanced for more robust interpretation of `pandas_ta`'s outputs, standardizing float representations within expected column names for unwavering accuracy.

2.  **The Illuminated Neon Interface: A Symphony of Light:**
    *   Fulfilling your vision, the `NEON` color dictionary has been significantly expanded, its vibrant energies now suffusing the script's entire console manifestation.
    *   **Systematic Luminescence - Clarity Through Color:**
        *   **Operational Aura:** `NEON.INFO`, `NEON.DEBUG`, `NEON.WARNING`, `NEON.ERROR`, `NEON.CRITICAL`, and `NEON.SUCCESS` now clearly delineate log levels and critical operational feedback with distinct hues.
        *   **Strategic Whispers:** `NEON.STRATEGY` distinctly highlights messages emanating from the core strategy logic.
        *   **Runic Inscriptions:** `NEON.PARAM` and `NEON.VALUE` make configuration parameters and their settings shine with clarity.
        *   **Treasury's Glow:** `NEON.PRICE`, `NEON.QTY`, and `NEON.PNL_POS/NEG/ZERO` bathe financial figures in an intuitive light, enhancing comprehension of profits, losses, and quantities.
        *   **Positional Beacons:** `NEON.SIDE_LONG/SHORT/FLAT` provide immediate, unambiguous visual cues for the bot's current market stance.
        *   **Scroll Structure:** `NEON.HEADING` and `NEON.SUBHEADING` organize the flow of information with thematic emphasis.
        *   **Commanding Presence:** `NEON.ACTION` brightly underscores crucial operational commands like "Placing Order" or "Closing Position."
    *   **Enhanced Log Formatting:** The `_format_for_log` helper function now masterfully wields these expanded colors, painting boolean trends with more evocative descriptions like "Upward Flow" and "Downward Tide."
    *   **Judicious Brilliance:** `Style.BRIGHT` is artfully applied to draw the eye to the most pivotal information, ensuring no critical detail is lost in the shadows.

3.  **Scroll Identity & Memory Glyphs (Version and State Management):**
    *   The scroll now proudly bears the mark of **`v2.8.0`**.
    *   Its memory glyph, the `STATE_FILE_NAME`, has been re-inscribed as `pyrmethus_phoenix_state_v280.json`, ensuring continuity with this new iteration.

4.  **Synergy with Ancient Wards (Integration with Existing Advanced Features):**
    *   These new enchantments—the `DualSupertrendMomentumStrategy` and the enhanced Neon Illumination—are not mere additions but are intricately woven into the sophisticated framework of v2.6.1. The established powers of persistence, dynamic risk/SL calculations, pyramiding architecture, and partial close logic now seamlessly integrate with, and are driven by, the signals from this new strategic core.

All pre-existing sophisticated enchantments—including state persistence through the Phoenix Feather, dynamic ATR stop-loss wards, intricate pyramiding logic, and detailed trade metrics—have been meticulously preserved. They now operate in concert with the new strategy, guided by its signals and presented with unparalleled clarity through the Strategic Illumination interface.
