import sqlite3
import pandas as pd

def list_tables(db_path: str) -> list[str]:
    """
    Return a list of all table names in a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    list[str]
        List of table names.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name;
    """)

    tables = [row[0] for row in cursor.fetchall()]

    conn.close()
    return tables


def list_columns(db_path: str, table_name: str) -> list[str]:
    """
    Return a list of column names for a table in a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    table_name : str
        Name of the table.

    Returns
    -------
    list[str]
        List of column names.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"PRAGMA table_info('{table_name}');")
    columns = [row[1] for row in cursor.fetchall()]

    conn.close()
    return columns


def get_table_schema(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Return the schema of a SQLite table as a pandas DataFrame.

    Columns returned:
        cid, name, type, notnull, default_value, primary_key
    """
    with sqlite3.connect(db_path) as conn:
        schema = pd.read_sql_query(
            f"PRAGMA table_info('{table_name}')",
            conn,
        )

    return schema


def load_table(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Load a SQLite table into a pandas DataFrame.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    table_name : str
        Name of the table to load.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all rows from the table.
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn)

    return df
