import plotly.express as px
import pandas as pd


def confidence_chart(confidence):

    df = pd.DataFrame({
        "Metric": ["Confidence"],
        "Value": [confidence]
    })

    fig = px.bar(
        df,
        x="Metric",
        y="Value",
        title="Prediction Confidence"
    )

    return fig