import datetime

import pandas as pd
import streamlit as st
import timeago
import plotly.graph_objects as go


st.set_page_config(layout="wide")
now = datetime.datetime.now()

MATCH_RESULTS_URL = "https://huggingface.co/datasets/huggingface-projects/bot-fight-data/raw/main/soccer_history.csv"


@st.cache_data(ttl=1800)
def fetch_match_history():
    """
    Fetch match history.
    Cache the result for 30min to avoid unnecessary requests.
    Return a DataFrame.
    """
    df = pd.read_csv(MATCH_RESULTS_URL)
    df["timestamp"] = pd.to_datetime(df.timestamp, unit="s")
    df.columns = ["home", "away", "timestamp", "result"]
    return df


def days_left():
    end_date = datetime.date(2023, 4, 30)
    today = datetime.date.today()
    time_until_date = end_date - today
    return time_until_date.days


def num_matches_played():
    return match_df.shape[0]


match_df = fetch_match_history()
teams = sorted(list(pd.concat([match_df["home"], match_df["away"]]).unique()), key=str.casefold)

st.title("SoccerTwos Challenge Analytics")

team_results = {}
for i, row in match_df.iterrows():
    home_team = row["home"]
    away_team = row["away"]
    result = row["result"]

    if home_team not in team_results:
        team_results[home_team] = [0, 0, 0]

    if away_team not in team_results:
        team_results[away_team] = [0, 0, 0]

    if result == 0:
        team_results[home_team][2] += 1
        team_results[away_team][0] += 1
    elif result == 1:
        team_results[home_team][0] += 1
        team_results[away_team][2] += 1
    else:
        team_results[home_team][1] += 1
        team_results[away_team][1] += 1


df = pd.DataFrame.from_dict(
    team_results, orient="index", columns=["wins", "draws", "losses"]
).sort_index()
df[["owner", "team"]] = df.index.to_series().str.split("/", expand=True)
df = df[["owner", "team", "wins", "draws", "losses"]]
df["win_pct"] = (df["wins"] / (df["wins"] + df["draws"] + df["losses"])) * 100

stats = df

tab_team, tab_competition = st.tabs(["Results", "Competition stats"])


def get_text_result(row, team_name):
    if row["home"] == team_name:
        if row["result"] == 1:
            return "Win"
        elif row["result"] == 0.5:
            return "Draw"
        else:
            return "Loss"
    elif row["away"] == team_name:
        if row["result"] == 0:
            return "Win"
        elif row["result"] == 0.5:
            return "Draw"
        else:
            return "Loss"


with tab_team:

    team = st.selectbox("Team", teams)

    col1, col2 = st.columns(2)

    with col1:


        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Wins", f"{stats.loc[[team]]['wins'][0]}")
        with c2:
            st.metric("Draws", f"{stats.loc[[team]]['draws'][0]}")
        with c3:
            st.metric("Losses", f"{stats.loc[[team]]['losses'][0]}")

        st.write("Results")
        res_df = match_df[(match_df["home"] == team) | (match_df["away"] == team)]
        res_df["result"] = res_df.apply(lambda row: get_text_result(row, team), axis=1)
        opponent_column = res_df.apply(
            lambda row: row["away"] if row["home"] == team else row["home"], axis=1
        )
        res_df["vs"] = opponent_column
        result_column = res_df["result"]
        new_df = pd.concat([opponent_column, result_column], axis=1)
        new_df.columns = ["vs", "result"]
        res_df[["owner", "team"]] = res_df["vs"].str.split("/", expand=True)
        res_df["played"] = res_df["timestamp"].apply(lambda x: timeago.format(x, now))
        res_df.sort_values(by=["timestamp"], ascending=False, inplace=True)
        disp_res_df = res_df.drop(["home", "away", "vs", "timestamp"], axis=1)

        def highlight_wins(s):
            colour = {
                "Win": "LightGreen",
                "Draw": "LightYellow",
                "Loss": "LightSalmon",
            }
            return [f"background-color: {colour[s.result]}"] * len(s)

        st.dataframe(disp_res_df.style.apply(highlight_wins, axis=1))

    with col2:

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Win rate", f"{stats.loc[[team]]['win_pct'][0]:.2f}%")

            joined = res_df["timestamp"].min()
        with c2:
            st.metric("Competing since", f"{timeago.format(joined, now)}")

        grouped = res_df.groupby([res_df["timestamp"].dt.date, "result"]).size().reset_index(name="count")

        loss_trace = go.Bar(
            x=grouped.loc[grouped["result"] == "Loss", "timestamp"],
            y=grouped.loc[grouped["result"] == "Loss", "count"],
            name="Losses",
            marker=dict(color='red')
        )
        draw_trace = go.Bar(
            x=grouped.loc[grouped["result"] == "Draw", "timestamp"],
            y=grouped.loc[grouped["result"] == "Draw", "count"],
            name="Draws",
            marker=dict(color='orange')
        )
        win_trace = go.Bar(
            x=grouped.loc[grouped["result"] == "Win", "timestamp"],
            y=grouped.loc[grouped["result"] == "Win", "count"],
            name="Wins",
            marker=dict(color='green')
        )

        fig = go.Figure(data=[win_trace, draw_trace, loss_trace])
        fig.update_layout(barmode="stack")
        st.plotly_chart(fig)


with tab_competition:
    col1, col2, col3 = st.columns(3)

    col1.metric("Matches played", f"{num_matches_played():,d}")
    col2.metric("Live models", f"{len(teams)}")
    col3.metric("Season ends in", f"{days_left()} days")

    match_counts = (
        match_df.groupby(match_df["timestamp"].dt.date).size().reset_index(name="count")
    )
    match_counts["matches_played"] = match_counts["count"].cumsum()

    st.title("Matches played")
    st.area_chart(match_counts.set_index("timestamp")["matches_played"])

