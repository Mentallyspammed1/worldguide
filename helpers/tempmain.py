# main.py - Temporary Test
async def main():
    global logger, exchange_instance_ref
    # ... (logger setup as before) ...
    # ... (load config as before) ...
    logger.info("--- ISOLATED INITIALIZATION TEST ---")
    test_exchange = None
    try:
        test_exchange = await bybit.initialize_bybit(
            full_config["API_CONFIG"]
        )  # Pass only API config
        if test_exchange:
            logger.success("Isolated initialize_bybit succeeded!")
            exchange_instance_ref = test_exchange  # Store for cleanup
        else:
            logger.error("Isolated initialize_bybit failed (returned None).")
    except Exception as e:
        logger.error(
            f"Isolated initialize_bybit failed with exception: {e}", exc_info=True
        )
    finally:
        # Cleanup
        if (
            exchange_instance_ref
            and hasattr(exchange_instance_ref, "close")
            and callable(exchange_instance_ref.close)
        ):
            if not exchange_instance_ref.closed:
                logger.info("Closing test exchange instance...")
                await exchange_instance_ref.close()
