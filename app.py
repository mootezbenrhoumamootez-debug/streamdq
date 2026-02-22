import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from solution import preprocess, load_model, predict

st.set_page_config(page_title="InsureIQ — Bundle Recommender", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
  .main { background-color: #f5f2ec; }
  .block-container { padding-top: 2rem; }
  .bundle-card { background: #0f0f0f; color: #f5f2ec; border-radius: 8px; padding: 2rem; text-align: center; margin-top: 1rem; }
  .bundle-number { font-size: 4rem; font-weight: 900; color: #c9a84c; line-height: 1; }
  .bundle-name { font-size: 1.4rem; font-weight: 600; margin-top: 0.5rem; }
  .bundle-label { font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #c9a84c; margin-bottom: 0.5rem; }
  .stButton > button { background-color: #0f0f0f !important; color: white !important; border: none !important; padding: 0.75rem 2rem !important; font-size: 1rem !important; font-weight: 600 !important; width: 100% !important; border-radius: 4px !important; }
  .stButton > button:hover { background-color: #3d4a5c !important; }
  .agent-box { background: #FFFFFF; border-left: 4px solid #3b82f6; border-radius: 4px; padding: 1rem 1.2rem; margin-top: 1rem; font-size: 0.95rem; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

BUNDLE_NAMES = {
    0: "Auto Comprehensive", 1: "Auto Liability Basic", 2: "Basic Health",
    3: "Family Comprehensive", 4: "Health Dental Vision", 5: "Home Premium",
    6: "Home Standard", 7: "Premium Health Life", 8: "Renter Basic", 9: "Renter Premium",
}

BUNDLE_DESCRIPTIONS = {
    0: "Covers damage to the customer's own vehicle from accidents, theft, weather, and other incidents.",
    1: "Covers damage the customer causes to others' vehicles or property — basic legal minimum.",
    2: "Essential health coverage for individuals: doctor visits, hospital stays, and emergencies.",
    3: "Comprehensive health and life coverage designed for households with multiple dependents.",
    4: "Health plan bundled with dental and vision — ideal for employed families.",
    5: "Premium homeowners policy covering structure, contents, liability, and natural disasters.",
    6: "Standard homeowners coverage for structure and contents at an affordable price point.",
    7: "Top-tier health insurance combined with life insurance for maximum protection.",
    8: "Basic renters insurance covering personal belongings and liability for tenants.",
    9: "Enhanced renters insurance with higher coverage limits and additional protections.",
}

@st.cache_resource
def get_model():
    return load_model()

model = get_model()


def explain_prediction(bundle_id, bundle_name, profile: dict):
    api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️ Set your `GROQ_API_KEY` in Streamlit secrets to enable AI explanations."

    from groq import Groq
    client = Groq(api_key=api_key)

    profile_summary = "\n".join([
        f"- Annual Income: ${profile['income']:,.0f}",
        f"- Employment: {profile['employment'] or 'Not specified'}",
        f"- Adult Dependents: {profile['adult_dep']}, Children: {profile['child_dep']}, Infants: {profile['infant_dep']}",
        f"- Vehicles on Policy: {profile['vehicles']}",
        f"- Existing Policyholder: {profile['existing']}",
        f"- Previous Claims: {profile['claims_filed']}",
        f"- Years Without Claims: {profile['years_no_claim']}",
        f"- Deductible Tier: {profile['deductible'] or 'Not specified'}",
        f"- Payment Schedule: {profile['payment'] or 'Not specified'}",
        f"- Acquisition Channel: {profile['channel'] or 'Not specified'}",
    ])

    prompt = f"""You are an expert insurance advisor. A machine learning model has recommended the following bundle for a customer:

Bundle ID: {bundle_id}
Bundle Name: {bundle_name}
Bundle Description: {BUNDLE_DESCRIPTIONS[bundle_id]}

Customer Profile:
{profile_summary}

In 3-4 sentences, explain to the customer in plain, friendly language WHY this bundle is a good fit for their profile. Be specific — reference their actual profile details. Do not mention the ML model. Sound like a knowledgeable human advisor."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


st.title("🛡️ InsureIQ — Bundle Recommender")
st.markdown("Enter a customer profile to get an **AI-powered insurance bundle recommendation**.")
st.divider()

col_form, col_result = st.columns([2, 1], gap="large")

with col_form:
    st.subheader("Customer Profile")

    st.markdown("**Demographics & Financials**")
    c1, c2 = st.columns(2)
    income         = c1.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    employment     = c2.selectbox("Employment Status", ["", "Employed_FullTime", "Employed_PartTime", "Self_Employed", "Unemployed", "Retired"])
    adult_dep      = c1.number_input("Adult Dependents", min_value=0, value=1, step=1)
    child_dep      = c2.number_input("Child Dependents", min_value=0, value=0, step=1)
    infant_dep     = c1.number_input("Infant Dependents", min_value=0, value=0, step=1)
    region         = c2.text_input("Region Code", placeholder="e.g. AUT")

    st.markdown("**Policy Preferences**")
    c3, c4 = st.columns(2)
    deductible     = c3.selectbox("Deductible Tier", ["", "Tier_1_High_Ded", "Tier_2_Mid_Ded", "Tier_3_Low_Ded", "Tier_4_Zero_Ded"])
    payment        = c4.selectbox("Payment Schedule", ["", "Monthly_EFT", "Quarterly", "Semi_Annual", "Annual"])
    vehicles       = c3.number_input("Vehicles on Policy", min_value=0, value=0, step=1)
    riders         = c4.number_input("Custom Riders Requested", min_value=0, value=0, step=1)
    channel        = c3.selectbox("Acquisition Channel", ["", "Direct_Website", "Aggregator_Site", "Broker_Referral", "Employer_Group", "Phone_Call"])
    broker_type    = c4.selectbox("Broker Agency Type", ["", "Urban_Boutique", "National_Corporate", "Regional_Mid_Size", "Online_Only"])

    st.markdown("**Customer History**")
    c5, c6 = st.columns(2)
    existing       = c5.selectbox("Existing Policyholder", ["No", "Yes"])
    claims_filed   = c6.number_input("Previous Claims Filed", min_value=0, value=0, step=1)
    years_no_claim = c5.number_input("Years Without Claims", min_value=0, value=0, step=1)
    prev_duration  = c6.number_input("Prior Policy Duration (months)", min_value=0, value=0, step=1)

    st.markdown("**Policy Timeline**")
    c7, c8, c9 = st.columns(3)
    start_year     = c7.number_input("Start Year", min_value=2010, max_value=2025, value=2020, step=1)
    start_month    = c8.selectbox("Start Month", ["January","February","March","April","May","June","July","August","September","October","November","December"])
    start_day      = c9.number_input("Start Day", min_value=1, max_value=31, value=1, step=1)
    days_quote     = st.number_input("Days Since Quote", min_value=0, value=0, step=1)

    predict_btn = st.button("Get Recommendation →")

with col_result:
    st.subheader("Recommendation")

    if predict_btn:
        row = {
            "User_ID": "USR_000000",
            "Estimated_Annual_Income": income,
            "Employment_Status": employment or None,
            "Adult_Dependents": adult_dep,
            "Child_Dependents": float(child_dep),
            "Infant_Dependents": infant_dep,
            "Region_Code": region or None,
            "Deductible_Tier": deductible or None,
            "Payment_Schedule": payment or None,
            "Vehicles_on_Policy": vehicles,
            "Custom_Riders_Requested": riders,
            "Acquisition_Channel": channel or None,
            "Broker_Agency_Type": broker_type or None,
            "Existing_Policyholder": 1 if existing == "Yes" else 0,
            "Previous_Claims_Filed": claims_filed,
            "Years_Without_Claims": years_no_claim,
            "Previous_Policy_Duration_Months": prev_duration,
            "Policy_Start_Year": start_year,
            "Policy_Start_Month": start_month,
            "Policy_Start_Day": start_day,
            "Days_Since_Quote": days_quote,
            "Policy_Cancelled_Post_Purchase": 0,
            "Policy_Start_Week": 1,
            "Grace_Period_Extensions": 0,
            "Policy_Amendments_Count": 0,
            "Broker_ID": None,
            "Employer_ID": None,
            "Underwriting_Processing_Days": 0,
        }

        df = pd.DataFrame([row])
        df_processed = preprocess(df)
        result = predict(df_processed, model)
        bundle_id = int(result["Purchased_Coverage_Bundle"].iloc[0])
        bundle_name = BUNDLE_NAMES[bundle_id]

        st.markdown(f"""
        <div class="bundle-card">
          <div class="bundle-label">Recommended Bundle</div>
          <div class="bundle-number">{bundle_id}</div>
          <div class="bundle-name">{bundle_name}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**🤖 Why this bundle?**")
        with st.spinner("Generating explanation..."):
            profile = dict(
                income=income, employment=employment, adult_dep=adult_dep,
                child_dep=child_dep, infant_dep=infant_dep, vehicles=vehicles,
                existing=existing, claims_filed=claims_filed,
                years_no_claim=years_no_claim, deductible=deductible,
                payment=payment, channel=channel,
            )
            explanation = explain_prediction(bundle_id, bundle_name, profile)

        st.markdown(f'<div class="agent-box">{explanation}</div>', unsafe_allow_html=True)

    else:
        st.info("Fill in the form and click **Get Recommendation**.")

    st.divider()
    st.markdown("**All Bundles**")
    for bid, bname in BUNDLE_NAMES.items():
        st.markdown(f"`{bid}` {bname}")
