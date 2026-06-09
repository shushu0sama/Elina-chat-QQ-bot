"""Pytest configuration — initializes NoneBot before test collection."""
import nonebot

# Must run at module level, before any test files are imported
nonebot.init(_env_file=".env.example")
