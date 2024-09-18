from pathlib import Path

import pandas as pd

# Player;Pos;Age;Tm;G;GS;MP;FG;FGA;FG%;3P;3PA;3P%;2P;2PA;2P%;eFG%;FT;FTA;FT%;ORB;DRB;TRB;AST;STL;BLK;TOV;PF;PTS;Performance
position_mapping: dict[str, int] = {}


def load_data_from_csv():
    path = Path(__file__).parent.parent / "data" / "nba_dados.csv"
    print(path)
    return pd.read_csv(path)


def normalize_position(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Pos"], dtype=int, dummy_na=False)
    return data


def normalize_team(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Tm"], dtype=int, dummy_na=False)
    return data


def normalize_performance(data: pd.DataFrame):
    data = pd.get_dummies(data, columns=["Performance"], dtype=int, dummy_na=False)
    return data


if __name__ == "__main__":
    data = load_data_from_csv()
    data = data.drop(columns=["Player"])
    data = normalize_position(data)
    data = normalize_team(data)
    data = normalize_performance(data)
    print(data.head())
