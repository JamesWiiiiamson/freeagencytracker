from sqlalchemy import create_engine, text

from config import DB_URL


def main() -> None:
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version();")).scalar_one()
    print(version)


if __name__ == "__main__":
    main()
