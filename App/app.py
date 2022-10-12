#Core Pkgs
import streamlit as st
import tensorflow as tf

import io
import cv2
from PIL import Image, ImageOps

#EDA Pkgs
import pandas as pd
import numpy as np
import altair as alt

#load model
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('pireps_classifier.model')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

#Prediction function
class_names=['A/I PROBLEM-ENG', 'A/I PROBLEM-SYS', 'A/I PROBLEM-WING', 'ACARS',
       'ACCESS DOOR/PANEL', 'ACCESS PANEL', 'ADC PROBLEM', 'ADF', 'ADS-B',
       'AHRS', 'AILERON', 'AIR DISTRIBUTION', 'AIR DISTRIBUTION ',
       'AIR DISTRIBUTION?', 'AIR/GND FAIL', 'AOA', 'AP/YD', 'AP/YD?',
       'APU BLEED', 'APU FAIL', 'APU GEN PROBLEM', 'APU PROBLEM',
       'AURAL WARNING', 'BAGGAGE DOOR', 'BATT OFF BUS', 'BATT PROBLEM',
       'BATT PROBLEM ', 'BIRD STRIKE', 'BIRDSTRIKE', 'BLEED LEAK',
       'BLEED LOW TEMP', 'BLEED OVER TEMP', 'BLEED OVERTEMP',
       'BLEED VALVE', 'BRAKE DEGRADE', 'BRAKE GRABBING/DRAGGING',
       'BRAKE PROBLEM', 'BRAKE PROBLEM?', 'BRAKE TEMP', 'BRAKE WORN',
       'BRIEFING CARDS', 'CABIN INTERIOR', 'CABIN INTERIOR?',
       'CARGO INTERIOR', 'CARGO INTERIOR?', 'CD PLAYER', 'CHECKLIST',
       'CIRCUIT BREAKER', 'CIRCUIT BREAKER ', 'CLEAR ICE DET', 'CLOCK',
       'COCKPIT DOOR', 'COCKPIT EQUIP/INTERIOR',
       'COCKPIT LIGHT/PUSHBUTTON', 'COFFEE POT/MAKER', 'COMM', 'COMM?',
       'COMPASS', 'COWLING', 'CREW ERROR', 'CREW O2', 'CREW O2 MASK',
       'CREW SEAT', 'CREW/OPS', 'CROSS FEED', 'CROSSBLEED', 'DAMAGE',
       'DAU', 'DEMO EQUIP', 'DENT', 'DFDR', 'DOOR INDICATION',
       'DOOR SHROUD', 'DU/PFD/MFD', 'DV WINDOW', 'EGPWS/WINDSHEAR/TERR',
       'EICAS', 'EIE', 'ELECTRICAL PROBLEM', 'ELECTRICAL PROBLEM?',
       'ELEVATOR PROBLEM', 'EMERGENCY EQUIPMENT', 'EMERGENCY EXIT DOOR',
       'EMERGENCY EXIT ROW WINDOW', 'EMERGENCY LIGHT', 'ENG CONTROL',
       'ENG EXCEEDANCE', 'ENG FUEL', 'ENG FUEL IMP BYPASS',
       'ENG FUEL LOW PRESSURE', 'ENG HIGH ITT', 'ENG IDLE STOP',
       'ENG IGNITION', 'ENG NO DISP/SHORT DISP', 'ENG OIL',
       'ENG OIL DEBRIS', 'ENG OIL PRESSURE', 'ENG OIL TEMP',
       'ENG SIGHT GLASS', 'ENGINE', 'ENGINE STARTING', 'ESSENTIAL POWER',
       'EXTERIOR LIGHT', 'EXTERNAL POWER', 'FA CALL SYS', 'FA SEAT',
       'FADEC', 'FDAU', 'FIRE DETECTION', 'FIRE EXTINGUISHING',
       'FLAP EXCEEDANCE/OVER SPEED (CREW ISSUE)', 'FLAP FAIL',
       'FLAP LOW SPEED', 'FLIGHT CONTROLS', 'FMS', 'FOD', 'FOD ',
       'FUEL CAP', 'FUEL IMBALANCE', 'FUEL LEAK', 'FUEL PUMP',
       'FUEL QTY IND', 'FUEL TANK', 'FUEL TRANSFER', 'FUEL VALVE',
       'FUEL VENT', 'GALLEY', 'GEAR DOOR', 'GEAR INDICATION/ACTUATION',
       'GEAR NOISE', 'GEAR STRUT', 'GEAR VIBRATION/SHIMMY', 'GEN OFF BUS',
       'GROUND DAMAGE', 'GUST LOCK', 'GUST LOCK (CREW ISSUE)', 'HYD PUMP',
       'HYD SERVICE', 'HYD SYS', 'HYD SYS LOW/LEAK', 'IC-600',
       'ICE DETECTION', 'INTERIOR LIGHT', 'ISIS', 'KNOB/BEZEL',
       'KNOB/BEZEL ', 'LANDING GEAR DOWNLOCK', 'LANDING GEAR UPLOCK',
       'LAV DOOR', 'LAV INTERIOR', 'LAV LIGHT', 'LAV PROBLEM',
       'LEAK/WINDOW/DOORS/SEALS', 'LIGHTNING STRIKE', 'LOGBOOK ISSUE',
       'MAIN GEN PROBLEM', 'MASK DROP', 'MISSING LOOSE/SCREW',
       'MISSING/LOOSE HARDWARE', 'MISSING/LOOSE SCREW', 'MX INDUCED',
       'NACELLES/PYLONS', 'NAV UNIT PROBLEM', 'NAV UNIT PROBLEM?',
       'NOISE', 'NOISE?', 'NOSE WHEEL STEERING', 'O2 MASK ',
       'O2 RELEASE TOOL', 'O2 SERVICE', 'OIL SERVICE', 'OTHER', 'OTHER ',
       'OTHER?', 'OVHD BIN', 'PA/INTERPHONE', 'PACK ACM', 'PACK DUCT',
       'PACK INOP/FAIL', 'PACK NOISE', 'PACK OVERLOAD', 'PACK TEMP',
       'PACK TEMP?', 'PACK VALVE', 'PAPERWORK', 'PARKING BRAKE',
       'PAX INDUCED', 'PAX O2 MASK', 'PAX SEAT', 'PAX USED', 'PAX WINDOW',
       'PAX-SEAT BELT EXTENDER', 'PAX/CREW DOOR', 'PBE', 'PBE ',
       'PITCH TRIM', 'PITOT/STATIC TUBE', 'PLACARD', 'PORTABLE O2',
       'POTABLE WATER', 'POTABLE WATER?', 'PRESSURE REFUEL',
       'PRESSURIZATION PROBLEM', 'PSU', 'RADIO ALT', 'RMU', 'ROLL TRIM',
       'RUDDER SYS', 'RVSM', 'SCHEDULED', 'SEALANT', 'SEAT TRACK',
       'SENSOR HEATING', 'SERVICE DOOR', 'SHOCK ABSORBERS',
       'SMOKE/SMELL IN COCKPIT/CABIN', 'SPECIAL INSPECTION', 'SPEED TAPE',
       'SPOILER PROBLEM', 'SPS', 'STATIC WICK/BONDING STRAP',
       'STICK PUSHER', 'T/O CONFIG', 'T/R DISAGREE', 'T/R PROBLEM',
       'TBCH', 'TCAS', 'TIRE CUT/PUNCTURE', 'TIRE PROBLEM', 'TIRE WORN',
       'TOWBAR', 'TRANSPONDER', 'VHF', 'VOR/DME/LOC/ILS', 'WEATHER RADAR',
       'WINDOW', 'WINDSHIELD', 'WING', 'WING ', 'WS HEAT', 'WS SEALANT',
       'WS WIPER', 'YAW TRIM', 'YOKE']

def predict_pireps(action):
    pred_prob=tf.squeeze(model.predict(action))
    pred=tf.argmax(pred_prob)
    pred_class=class_names[pred]
    return pred_class

def pred_pireps(action):
    pred_prob=tf.squeeze(model.predict(action)).numpy()
    #pred=tf.argmax(pred_prob).numpy()
    return pred_prob

#Main application

def main():
    st.title("Pireps Classifier App")
    menu=["Home", "Monitor", "About"]
    choice=st.sidebar.selectbox("Menu", menu)

    if choice =="Home":
        st.subheader("Home-Pireps Actions")

        with st.form(key='Pireps_clf_form'):
            raw_text=st.text_area("Type Here")
            submit_text=st.form_submit_button(label="Submit")

        if submit_text:
            col1, col2=st.columns(2)

            #Apply the functions Here
            prediction=predict_pireps([raw_text])
            prediction_probability=pred_pireps([raw_text])


            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                st.write(prediction)
                st.write("Confidence:{}".format(np.max(prediction_probability)))

            with col2:
                st.success("Prediction Probability")
                #st.write(prediction_probability)
                proba_df=pd.DataFrame([prediction_probability], columns=class_names)
                proba_df=proba_df.sort_index(axis=1)
                #st.write(proba_df.T)
                proba_df_clean=proba_df.T.reset_index()
                proba_df_clean.columns=['category','probability']
                proba_df_clean=proba_df_clean.sort_values("probability", ascending=False)[:5]
                st.dataframe(proba_df_clean.sort_values("probability", ascending=False))
               
                fig =alt.Chart(proba_df_clean).mark_bar().encode(
                    x='category', 
                    y=alt.Y('probability', sort="-x"))
                st.altair_chart(fig, use_container_width=True)
                


    elif choice=="Monitor":
        st.subheader("Monitor App")

    else:
        st.subheader("About")


if __name__=='__main__':
    main()
