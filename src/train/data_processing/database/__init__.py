# File: src/train/data_processing/database/__init__.py
# 数据库模块 / Database module

from .database_setup import setup_database, get_connection

__all__ = [
    'setup_database',
    'get_connection'
]
