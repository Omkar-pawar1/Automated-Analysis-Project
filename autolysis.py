import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import traceback

# Ensure environment variable for AI Proxy token is set
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable not set.")
    sys.exit(1)

url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def analyze_csv(filename):
    try:
        encodings = ["utf-8", "ISO-8859-1", "cp1252"]
        for encoding in encodings:
            try:
                df = pd.read_csv(filename, encoding=encoding, low_memory=False)
                break
            except UnicodeDecodeError:
                continue
        else:
            print(f"Error: Unable to decode the file with common encodings {encodings}.")
            sys.exit(1)

        if df.columns[0] == df.iloc[0].values[0]:
            print("Warning: No headers detected, assigning default column names.")
            df = pd.read_csv(filename, header=None, encoding=encoding, low_memory=False)
            df.columns = [f"Column_{i}" for i in range(len(df.columns))]

        if df.shape[1] == 1:
            print("Warning: The file may have a non-standard delimiter. Trying semicolon (;).")
            df = pd.read_csv(filename, delimiter=";", encoding=encoding, low_memory=False)

        df.columns = df.columns.str.strip()

        if df.columns.duplicated().any():
            print("Warning: Duplicate column names detected. Renaming duplicates.")
            df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)


        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
        }

        if df.empty:
            print("Error: The CSV file is empty.")
            sys.exit(1)

        return df, summary

    except Exception as e:
        print(f"Unexpected error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

def query_llm_for_charts(summary):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        
        prompt = (
            "You are a data visualization expert. Based on the following dataset summary, "
            "recommend only the top 2 suitable chart types for analyzing this dataset. "
            "For each chart, provide ONLY the Python code needed to generate it using matplotlib or seaborn, "
            "and ensure the code shows and saves the plot as a PNG file (e.g., `plt.savefig('chart.png')`) "
            "not only displaying it. Assume the dataset is loaded into a variable `df`.\n\n"
            f"Dataset Summary:\n{summary}\n\n"
            "Your response should be strictly in this format:\n"
            "{\n"
            "  \"charts\": [\n"
            "    {\"type\": \"Chart Type\", \"code\": \"Python code for chart\"},\n"
            "    {\"type\": \"Chart Type\", \"code\": \"Python code for chart\"}\n"
            "  ]\n"
            "}"
        )
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        recommendations = response_data['choices'][0]['message']['content']
        print("Chart recommendations received:", recommendations)
        return recommendations
    except Exception as e:
        print(f"Error querying LLM for chart recommendations: {e}\n{traceback.format_exc()}")
        sys.exit(1)


def generate_charts_from_llm(recommendations, df, output_dir):
    charts = []
    try:
        import json
        chart_data = json.loads(recommendations)
        for chart in chart_data['charts']:
            chart_type = chart['type']
            chart_code = chart['code']
            
            # Define the chart file path
            chart_filename = os.path.join(output_dir, f"{chart_type}.png")
            chart_code_with_save = chart_code.replace(
                "plt.savefig(", f"plt.savefig('{chart_filename}'"
            )
            
            # Validate and execute the updated code
            local_vars = {'df': df}
            print(f"Generating {chart_type} and saving to {chart_filename}...")
            exec(chart_code_with_save, globals(), local_vars)
            charts.append(chart_filename)  # Save the full path of the chart
            
    except json.JSONDecodeError as je:
        print(f"Error parsing LLM output: {je}\n{traceback.format_exc()}")
    except SyntaxError as se:
        print(f"Syntax error in chart code: {se}\n{traceback.format_exc()}")
    except Exception as e:
        print(f"Error executing chart code: {e}\n{traceback.format_exc()}")
    return charts




def query_llm_for_story(summary, charts):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}"
        }
        prompt = (
            "You are a data analyst and a storyteller. Based on the following dataset summary, "
            "generate a concise and insightful detailed story following this structure:\n\n"
            "1. **Discovery:** Context about the dataset and potential insights.\n"
            "2. **Analysis:** Insights based on the charts. Mention trends or anomalies.\n"
            "3. **Implications:** Actions or decisions based on the analysis.\n\n"
            "Charts Generated:\n"
            f"{charts}\n\n"
            "Dataset Summary:\n"
            f"{summary}\n\n"
            "Use a clear and concise tone for a non-technical audience."
        )
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()
        story = response_data['choices'][0]['message']['content']
        print(" INCIDE QUERY FOR STROY ::::   Generated storytelling narrative:\n", story)
        return story
    except Exception as e:
        print(f"Error querying LLM for storytelling: {e}\n{traceback.format_exc()}")
        sys.exit(1)


def create_story_readme(filename, story, charts):
    try:
        f_name = filename.rsplit(".csv", 1)[0]
        
        # Define the directory and README file path
        dir_path = f_name
        readme_file = os.path.join(dir_path, "README.md")
        
        # Ensure the directory exists
        os.makedirs(dir_path, exist_ok=True)
        
        with open(readme_file, "w") as f:
            f.write("# Data Analysis Story\n\n")
            f.write(story)
            f.write("\n\n## Visualizations\n")
            for chart in charts:
                # Convert to relative path for the README file
                relative_path = os.path.basename(chart)
                f.write(f"![Chart]({relative_path})\n")
    except Exception as e:
        print(f"Error writing storytelling README.md: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    df, summary = analyze_csv(input_file)
    
    # Ensure output directory exists
    output_dir = input_file.rsplit(".csv", 1)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate Chart Recommendations
    chart_recommendations = query_llm_for_charts(summary)
    charts = generate_charts_from_llm(chart_recommendations, df, output_dir)

    # Step 2: Generate Storytelling Narrative
    story = query_llm_for_story(summary, charts)

    # Step 3: Save Narrative and Charts in README.md
    create_story_readme(input_file, story, charts)
    print("Analysis complete. Storytelling narrative written to README.md and PNG charts saved.")
