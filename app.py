import streamlit as st
import pandas as pd
import os
import json
from openai import AzureOpenAI
from dotenv import load_dotenv
import re
from elasticsearch import Elasticsearch

# Load environment variables from .env file
load_dotenv()

# Define the model name
MODEL = "gpt-4o-mini"

# Initialize Azure OpenAI client
Azure_Client = AzureOpenAI(
    api_key=os.getenv("AZURE_API"),
    azure_endpoint=os.getenv("AZURE_BASE_URL"),
    api_version=os.getenv("AZURE_API_VERSION")
)

# Elasticsearch connection details from environment variables
ELASTICSEARCH_ENDPOINT = os.getenv('elasticsearchendpoint')
ELASTIC_API_KEY = os.getenv('elasticapikey')

# Initialize the Elasticsearch client
es = Elasticsearch(ELASTICSEARCH_ENDPOINT, api_key=ELASTIC_API_KEY)

def fetch_clinical_trial_data(vaccine_type):
    """
    Fetch the clinical trial data from Elasticsearch based on the vaccine type.
    """
    query = {
        "query": {
            "bool": {
                "should": [
                    {"match": {"Conditions": vaccine_type}},
                    {"match": {"Conditions_x": vaccine_type}},
                    {"match": {"Intervention_Type": "BIOLOGICAL"}},
                    {"match": {"Modality": "Vaccine"}}
                ]
            }
        },
    }
    
    response = es.search(index="matched_clinicaltrial", body=query)
    hits = response.get('hits', {}).get('hits', [])
    
    clinical_trial_data = []
    for hit in hits:
        clinical_trial_data.append(hit['_source'])
    
    return clinical_trial_data

def user_input(Input_df, user_query):
    """
    Prepare the prompt for the model. This function converts the DataFrame into a string to use as context.
    """
    context = Input_df.to_string(index=False)
    prompt = f"""
    This is Clinical Trial data for the Vaccine Adjuvant entered by the user: {user_query}.
    This is the context of the clinical trial data: {context}
    """
    
    return prompt

def get_response(prompt, user_query):
    """
    Constructs a system message based on the user query, then sends both system and user messages
    to the Azure OpenAI model and returns its response.
    """
    query_parts = user_query.split(" | ")
    if len(query_parts) < 5:
        raise ValueError("User query does not contain all required fields.")
    
    vaccine_type = query_parts[0].split(":")[1].strip()
    adjuvant_properties = query_parts[1].split(":")[1].strip()
    study_filter = query_parts[2].split(":")[1].strip()
    pipeline = query_parts[3].split(":")[1].strip()
    clinical_phase = query_parts[4].split(":")[1].strip()

    system_prompt = f"""
    You are an Immunologist and Vaccine Researcher. Your task is to find the adjuvants for the vaccine type along with its properties, endpoints, clinical phase, pipeline, and study status.
    You are given the context of the clinical trial data and the user query.
    INSTRUCTIONS:
    1. Read the user query and the context carefully.
    2. Find the adjuvants for the vaccine type {vaccine_type} along with its properties and endpoints {adjuvant_properties}, clinical phase {clinical_phase}, and study status {study_filter}.
    3. Calculate the score for the adjuvants based on the properties and endpoints (should be safe, non-toxic, and have high efficacy), clinical phase (with higher weight for phases like Phase 3 and Phase 4), and the pipeline/study status (completed or in-pipeline get higher scores).
    4. Provide the answer in JSON format with the following keys and make sure to keep the format consistent for all outputs:
       {{
           "Adjuvant Name": "Adjuvant Name_1",
           "Score": "Score_1",
           "Insights": "Insights_1"
       }},
       {{
           "Adjuvant Name": "Adjuvant Name_2",
           "Score": "Score_2",
           "Insights": "Insights_2"
       }}
    5. The score should be between 0 and 10, and only provide the top 5 adjuvants.
    6. If you are not sure, please say "not sure".
    7. remove the ```tags.
    
    Output format:
    1. Adjuvant Name: "Adjuvant Name"
    2. Score: "Score"
    3. Provides insights about the score and why the adjuvant will be effective for the vaccine type {vaccine_type}.
    """
    
    response = Azure_Client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt.strip()}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()

def main():
    st.title("VaccX")
    st.subheader("Vaccine Adjuvant Analysis")
    st.write("Enter the parameters for the clinical trial analysis below:")

    # Create two columns for organizing user input fields
    col1, col2 = st.columns(2)
    with col1:
        vaccine_type = st.text_input("Enter the Vaccine Type")
        adjuvant_prop = st.text_input("Enter the Adjuvant Properties")
    with col2:
        study_status = st.text_input("Enter the Study Filter")
        pipeline_status = st.text_input("Enter the Pipeline Status")
        clinical_phase = st.text_input("Enter the Clinical Phase")

    # Process input only when Submit is clicked
    if st.button("Submit"):
        # Build the user query string using the pipe symbol as the field delimiter
        user_query = (
            f"Vaccine Type: {vaccine_type} | "
            f"Adjuvant Properties: {adjuvant_prop} | "
            f"Study Filter: {study_status} | "
            f"Pipeline: {pipeline_status} | "
            f"Clinical Phase: {clinical_phase}"
        )
        
        # Fetch clinical trial data from Elasticsearch
        with st.spinner("Fetching clinical trial data..."):
            clinical_trial_data = fetch_clinical_trial_data(vaccine_type)
        
        if not clinical_trial_data:
            st.error("No clinical trial data found for the given vaccine type.")
            return
        
        clinical_trial_df = pd.DataFrame(clinical_trial_data)
        
        # Build the prompt for the model
        prompt = user_input(clinical_trial_df, user_query)
        
            # Get the response from the Azure OpenAI model
        with st.spinner("Querying the model..."):
            answer = get_response(prompt, user_query)
            print("Model response: ", answer)
        
        st.subheader("ADJUVANTS FOR VACCINE")
        try:
            # Clean the answer: remove markdown code block formatting if present
            answer_clean = answer.strip()
            if answer_clean.startswith("```"):
                parts = answer_clean.split("```")
                if len(parts) >= 3:
                    answer_clean = parts[1].strip()
                    # Remove any leading "json" language identifier if present
                    if answer_clean.lower().startswith("json"):
                        answer_clean = answer_clean.split("\n", 1)[1].strip()
            
            # First, try to parse the entire cleaned answer as JSON
            try:
                parsed = json.loads(answer_clean)
            except json.JSONDecodeError:
                # Fallback: Use regex to extract individual JSON objects (less ideal)

                json_objects = re.findall(r'\{.*?\}', answer_clean, re.DOTALL)
                if not json_objects:
                    raise ValueError("No JSON objects found in the model response.")
                parsed = [json.loads(obj) for obj in json_objects]
            
            # Normalize the parsed data:
            if isinstance(parsed, dict):
                # If the dictionary has a key that holds a list (e.g., "Adjuvants"), extract it;
                # otherwise, wrap the dict in a list.
                if "Adjuvants" in parsed and isinstance(parsed["Adjuvants"], list):
                    data = parsed["Adjuvants"]
                else:
                    data = [parsed]
            elif isinstance(parsed, list):
                data = parsed
            else:
                raise ValueError("Parsed JSON has an unexpected format.")
            
            # Create a DataFrame from the dynamic data
            df_output = pd.DataFrame(data)
            
            # Create a styled version of the DataFrame with text wrapping in the Insights column.
            styled_df = df_output.style.set_properties(
                subset=["Insights"],
                **{"white-space": "normal", "word-wrap": "break-word"}
            )
            
            # Define custom CSS for additional styling.
            custom_css = """
            <style>
            table { width: 100%; border-collapse: collapse; }
            th { font-weight: bold; text-align: left; padding: 8px; border-bottom: 2px solid #ddd; }
            td { padding: 8px; border-bottom: 1px solid #ddd; }
            </style>
            """
            
            # Generate the HTML from the styled DataFrame
            html_styled = styled_df.to_html()
            # Remove inline CSS produced by the Styler (the first <style>...</style> block)
            html_clean = re.sub(r'<style[^>]*>.*?</style>', '', html_styled, flags=re.DOTALL)
            
            # Render the final HTML with your custom CSS only
            st.markdown(custom_css + html_clean, unsafe_allow_html=True)

            
            # # Display the DataFrame dynamically (the output adapts automatically)
            # st.dataframe(df_output, use_container_width=True)
            
        except Exception as e:
            st.error("Error processing model response: " + str(e))
            st.text("Raw answer: " + answer)





if __name__ == "__main__":
    main()
