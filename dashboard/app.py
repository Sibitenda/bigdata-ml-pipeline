import os
import time
import glob
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Big Data ML Monitoring Dashboard",
    layout="wide"
)

st.title("Real-Time Big Data ML Monitoring Dashboard")

st.markdown(
    """
    This dashboard monitors:

    - Batch ML Training
    - Spark Distributed ML
    - Kafka Producer Events
    - Kafka Consumer Predictions
    - Micro-Batch Inference
    - Event-Driven Alerts
    """
)

# Dashboard auto refresh
refresh_rate = 5

# Layout sections
top_col1, top_col2, top_col3 = st.columns(3)

batch_section = st.container()

microbatch_section = st.container()

producer_section = st.container()

consumer_section = st.container()

alert_section = st.container()

chart_section = st.container()


while True:

    try:

        st.sidebar.header("System Status")

        st.sidebar.success("Dashboard Running")

        st.sidebar.info(
            f"Refresh Rate: {refresh_rate} seconds"
        )

         
        # BATCH ML RESULTS
         

        batch_metrics_path = (
            "data/results/"
            "batch_model_results.csv"
        )

        if os.path.exists(batch_metrics_path):

            batch_df = pd.read_csv(
                batch_metrics_path
            )

            with batch_section:

                st.subheader(
                    "Batch ML Results"
                )

                st.dataframe(batch_df)

                fig1, ax1 = plt.subplots()

                ax1.bar(
                    batch_df["model"],
                    batch_df["accuracy"]
                )

                ax1.set_title(
                    "Batch ML Accuracy"
                )

                ax1.set_ylabel(
                    "Accuracy"
                )

                st.pyplot(fig1)

        else:

            st.warning(
                "Batch ML results not found"
            )

         
        # MICRO-BATCH METRICS
         

        microbatch_metrics_path = (
            "data/kafka_predictions/"
            "kafka_metrics.csv"
        )

        if os.path.exists(
            microbatch_metrics_path
        ):

            metrics_df = pd.read_csv(
                microbatch_metrics_path
            )

            with microbatch_section:

                st.subheader(
                    "Spark Micro-Batch Metrics"
                )

                st.dataframe(metrics_df)

                fig2, ax2 = plt.subplots()

                ax2.plot(
                    metrics_df["batch"],
                    metrics_df["accuracy"]
                )

                ax2.set_title(
                    "Micro-Batch Accuracy"
                )

                ax2.set_xlabel(
                    "Batch"
                )

                ax2.set_ylabel(
                    "Accuracy"
                )

                st.pyplot(fig2)

        else:

            st.warning(
                "Micro-batch metrics not found"
            )

         
        # PRODUCER EVENTS
         

        producer_files = glob.glob(
            "data/producer_logs/*.csv"
        )

        if producer_files:

            latest_producer = max(
                producer_files,
                key=os.path.getctime
            )

            producer_df = pd.read_csv(
                latest_producer
            )

            with producer_section:

                st.subheader(
                    "Kafka Producer Events"
                )

                st.dataframe(
                    producer_df.tail(20)
                )

                st.metric(
                    "Total Producer Events",
                    len(producer_df)
                )

        else:

            st.warning(
                "Producer logs not found"
            )

         
        # CONSUMER PREDICTIONS
         

        prediction_files = glob.glob(
            "data/kafka_predictions/*.csv"
        )

        prediction_files = [

            f for f in prediction_files

            if "metrics" not in f

        ]

        if prediction_files:

            latest_predictions = max(
                prediction_files,
                key=os.path.getctime
            )

            pred_df = pd.read_csv(
                latest_predictions
            )

            with consumer_section:

                st.subheader(
                    "Kafka Consumer Predictions"
                )

                st.dataframe(
                    pred_df.tail(20)
                )

                prediction_counts = (
                    pred_df["prediction"]
                    .value_counts()
                )

                fig3, ax3 = plt.subplots()

                ax3.bar(

                    prediction_counts.index.astype(str),
                    prediction_counts.values

                )

                ax3.set_title(
                    "Prediction Distribution"
                )

                ax3.set_xlabel(
                    "Prediction Class"
                )

                ax3.set_ylabel(
                    "Count"
                )

                st.pyplot(fig3)

        else:

            st.warning(
                "Consumer predictions not found"
            )

         
        # ALERT MONITORING
         

        with alert_section:

            st.subheader(
                "Event-Driven Alert Monitoring"
            )

            if prediction_files:

                high_alerts = (

                    pred_df["prediction"] == 2

                ).sum()

                medium_alerts = (

                    pred_df["prediction"] == 1

                ).sum()

                low_alerts = (

                    pred_df["prediction"] == 0

                ).sum()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Low Risk",
                    low_alerts
                )

                col2.metric(
                    "Medium Risk",
                    medium_alerts
                )

                col3.metric(
                    "High Risk",
                    high_alerts
                )

                if high_alerts > 10:

                    st.error(
                        "HIGH ALERT: "
                        "Potential outbreak detected"
                    )

                elif medium_alerts > 10:

                    st.warning(
                        "WARNING: "
                        "Moderate outbreak activity"
                    )

                else:

                    st.success(
                        "System Stable"
                    )

         
        # LIVE TERMINAL STYLE STATUS
         

        with chart_section:

            st.subheader(
                "Live System Activity"
            )

            live_logs = []

            if producer_files:

                live_logs.append(
                    "Kafka Producer Running"
                )

            if prediction_files:

                live_logs.append(
                    "Kafka Consumer Running"
                )

            if os.path.exists(
                microbatch_metrics_path
            ):

                live_logs.append(
                    "Spark Micro-Batch Inference Active"
                )

            if os.path.exists(
                batch_metrics_path
            ):

                live_logs.append(
                    "Batch ML Results Available"
                )

            for log in live_logs:

                st.code(log)

        time.sleep(refresh_rate)

        st.rerun()

    except Exception as e:

        st.error(
            f"Dashboard Error:\n{e}"
        )

        time.sleep(refresh_rate)

        st.rerun()