
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. ตั้งค่าหน้าเว็บ
# ==========================================
st.set_page_config(
    page_title="Course Completion AI Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# 2. CSS ปรับแต่งหน้าตา
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9CA3AF;
        font-size: 16px;
        margin-bottom: 20px;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. โหลดโมเดลและ Scaler
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load("student_completion_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler


FEATURES = [
    "login_days",
    "video_hours",
    "quiz_average",
    "assignments_submitted",
    "forum_posts",
    "course_level_encoded"
]

LEVEL_MAPPING = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2
}


# ==========================================
# 4. ฟังก์ชันช่วยทำนาย
# ==========================================
def prepare_input(login_days, video_hours, quiz_average,
                  assignments_submitted, forum_posts, course_level):

    return pd.DataFrame([{
        "login_days": login_days,
        "video_hours": video_hours,
        "quiz_average": quiz_average,
        "assignments_submitted": assignments_submitted,
        "forum_posts": forum_posts,
        "course_level_encoded": LEVEL_MAPPING[course_level]
    }], columns=FEATURES)


def predict_completion(model, scaler, input_df):
    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)[0]

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "โมเดลนี้ไม่รองรับ predict_proba จึงไม่สามารถแสดงเปอร์เซ็นต์ได้"
        )

    probabilities = model.predict_proba(scaled_data)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "ไม่พบคลาส 1 ในโมเดล กรุณาตรวจสอบการเข้ารหัส completed"
        )

    completed_index = classes.index(1)
    completion_probability = float(probabilities[completed_index])

    not_completed_probability = 1 - completion_probability

    return (
        prediction,
        completion_probability,
        not_completed_probability
    )


# ==========================================
# 5. ส่วนหัว Dashboard
# ==========================================
st.markdown(
    '<div class="main-title">🎓 Course Completion AI Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'ระบบวิเคราะห์พฤติกรรมและทำนายโอกาสเรียนจบคอร์สออนไลน์ด้วย Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 6. โหลดโมเดลและแสดงแอป
# ==========================================
try:
    model, scaler = load_assets()

    # ------------------------------
    # Sidebar: กรอกข้อมูลผู้เรียน
    # ------------------------------
    st.sidebar.header("📝 ข้อมูลผู้เรียน")
    st.sidebar.caption("ปรับค่าข้อมูลเพื่อทดลองการทำนาย")

    login_days = st.sidebar.number_input(
        "จำนวนวันที่ล็อกอิน",
        min_value=0, max_value=30, value=15, step=1
    )

    video_hours = st.sidebar.number_input(
        "ชั่วโมงดูวิดีโอ",
        min_value=0.0, max_value=100.0, value=10.0, step=0.5
    )

    quiz_average = st.sidebar.number_input(
        "คะแนนควิซเฉลี่ย (%)",
        min_value=0.0, max_value=100.0, value=70.0, step=1.0
    )

    assignments_submitted = st.sidebar.number_input(
        "จำนวนการบ้านที่ส่ง",
        min_value=0, max_value=10, value=5, step=1
    )

    forum_posts = st.sidebar.number_input(
        "จำนวนโพสต์ในฟอรัม",
        min_value=0, max_value=50, value=2, step=1
    )

    course_level = st.sidebar.selectbox(
        "ระดับหลักสูตร",
        ["Beginner", "Intermediate", "Advanced"]
    )

    st.sidebar.divider()
    st.sidebar.info(
        "การทำนายขึ้นอยู่กับโมเดลที่ฝึกไว้ "
        "ผลลัพธ์ไม่ใช่การรับประกันว่าจะเรียนจบจริง"
    )

    # ------------------------------
    # เตรียมข้อมูลและทำนาย
    # ------------------------------
    input_df = prepare_input(
        login_days,
        video_hours,
        quiz_average,
        assignments_submitted,
        forum_posts,
        course_level
    )

    prediction, completion_prob, not_completed_prob = predict_completion(
        model, scaler, input_df
    )

    # ==========================================
    # 7. KPI Cards
    # ==========================================
    st.markdown("### 📊 ภาพรวมผลการทำนาย")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "โอกาสเรียนจบ",
        f"{completion_prob * 100:.2f}%"
    )

    k2.metric(
        "โอกาสเรียนไม่จบ",
        f"{not_completed_prob * 100:.2f}%"
    )

    k3.metric(
        "วันที่ล็อกอิน",
        f"{login_days} วัน"
    )

    k4.metric(
        "คะแนนควิซ",
        f"{quiz_average:.1f}%"
    )

    # แสดงผลการทำนาย
    st.markdown("### 🎯 ผลลัพธ์จากโมเดล")

    if prediction == 1:
        st.success(
            f"🟢 **ผลทำนาย: มีแนวโน้มเรียนจบ** "
            f"| โอกาสเรียนจบ {completion_prob * 100:.2f}%"
        )
    else:
        st.warning(
            f"🟠 **ผลทำนาย: มีแนวโน้มเรียนไม่จบ** "
            f"| โอกาสเรียนจบ {completion_prob * 100:.2f}%"
        )

    st.progress(completion_prob)
    st.caption(
        "แถบแสดงความน่าจะเป็นที่โมเดลประเมินว่าเรียนจบ "
        "ไม่ใช่คะแนนความสามารถของผู้เรียน"
    )

    st.divider()

    # ==========================================
    # 8. Tabs แบ่งข้อมูล
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 เปรียบเทียบหลักสูตร",
        "📚 พฤติกรรมการเรียน",
        "🧮 ความน่าจะเป็น",
        "🗂️ ข้อมูลที่ใช้ทำนาย"
    ])

    # ==========================================
    # TAB 1: เปรียบเทียบระดับหลักสูตร
    # ==========================================
    with tab1:
        st.subheader("เปรียบเทียบโอกาสเรียนจบตามระดับหลักสูตร")

        st.write(
            "กราฟนี้ใช้ข้อมูลพฤติกรรมการเรียนชุดเดียวกัน "
            "แล้วเปลี่ยนเฉพาะระดับหลักสูตร เพื่อเปรียบเทียบผลจากโมเดล"
        )

        comparison_rows = []

        for level, encoded_level in LEVEL_MAPPING.items():
            test_df = input_df.copy()
            test_df["course_level_encoded"] = encoded_level

            pred, prob, not_prob = predict_completion(
                model, scaler, test_df
            )

            comparison_rows.append({
                "ระดับหลักสูตร": level,
                "โอกาสเรียนจบ (%)": round(prob * 100, 2),
                "โอกาสเรียนไม่จบ (%)": round(not_prob * 100, 2),
                "ผลทำนาย": (
                    "เรียนจบ" if pred == 1 else "เรียนไม่จบ"
                )
            })

        comparison_df = pd.DataFrame(comparison_rows)

        chart_df = comparison_df.set_index("ระดับหลักสูตร")[
            ["โอกาสเรียนจบ (%)", "โอกาสเรียนไม่จบ (%)"]
        ]

        st.bar_chart(chart_df)

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "หมายเหตุ: กราฟแสดงผลจากโมเดล ไม่ได้ยืนยันว่า "
            "ผู้เรียนระดับใดจะเรียนจบจริงมากที่สุด"
        )

    # ==========================================
    # TAB 2: พฤติกรรมการเรียน
    # ==========================================
    with tab2:
        st.subheader("ข้อมูลพฤติกรรมของผู้เรียน")

        behavior_df = pd.DataFrame({
            "พฤติกรรม": [
                "วันล็อกอิน",
                "ชั่วโมงดูวิดีโอ",
                "คะแนนควิซ",
                "การบ้านที่ส่ง",
                "โพสต์ในฟอรัม"
            ],
            "ค่าที่กรอก": [
                login_days,
                video_hours,
                quiz_average,
                assignments_submitted,
                forum_posts
            ]
        })

        left, right = st.columns([1.3, 1])

        with left:
            st.markdown("**กราฟเปรียบเทียบค่าที่กรอก**")
            st.bar_chart(
                behavior_df.set_index("พฤติกรรม")
            )

        with right:
            st.markdown("**รายละเอียดข้อมูล**")
            st.dataframe(
                behavior_df,
                use_container_width=True,
                hide_index=True
            )

            st.markdown("**ระดับหลักสูตรที่เลือก**")
            st.info(
                f"{course_level} "
                f"(รหัส {LEVEL_MAPPING[course_level]})"
            )

        st.caption(
            "ค่าของแต่ละพฤติกรรมมีหน่วยและช่วงที่แตกต่างกัน "
            "กราฟนี้จึงใช้ดูค่าที่ป้อน ไม่ใช่การจัดอันดับความสำคัญของ Feature"
        )

    # ==========================================
    # TAB 3: ความน่าจะเป็น
    # ==========================================
    with tab3:
        st.subheader("ความน่าจะเป็นของผลลัพธ์")

        probability_df = pd.DataFrame({
            "ผลลัพธ์": ["เรียนจบ", "เรียนไม่จบ"],
            "ความน่าจะเป็น (%)": [
                round(completion_prob * 100, 2),
                round(not_completed_prob * 100, 2)
            ]
        })

        st.bar_chart(
            probability_df.set_index("ผลลัพธ์")
        )

        p1, p2 = st.columns(2)

        p1.metric(
            "เรียนจบ",
            f"{completion_prob * 100:.2f}%"
        )

        p2.metric(
            "เรียนไม่จบ",
            f"{not_completed_prob * 100:.2f}%"
        )

        st.caption(
            "เปอร์เซ็นต์มาจาก predict_proba ของโมเดล "
            "ความน่าเชื่อถือของเปอร์เซ็นต์ขึ้นอยู่กับคุณภาพข้อมูล "
            "และการประเมินโมเดล"
        )

    # ==========================================
    # TAB 4: ตารางข้อมูล
    # ==========================================
    with tab4:
        st.subheader("ข้อมูล Input ที่ส่งเข้าโมเดล")

        display_df = input_df.copy()
        display_df["course_level"] = course_level

        display_df = display_df[
            [
                "login_days",
                "video_hours",
                "quiz_average",
                "assignments_submitted",
                "forum_posts",
                "course_level",
                "course_level_encoded"
            ]
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("**รายละเอียด Feature**")

        feature_info = pd.DataFrame({
            "Feature": FEATURES,
            "ความหมาย": [
                "จำนวนวันที่ล็อกอิน",
                "ชั่วโมงดูวิดีโอ",
                "คะแนนควิซเฉลี่ย",
                "จำนวนการบ้านที่ส่ง",
                "จำนวนโพสต์ในฟอรัม",
                "รหัสระดับหลักสูตร"
            ]
        })

        st.dataframe(
            feature_info,
            use_container_width=True,
            hide_index=True
        )

    # ==========================================
    # 9. Footer
    # ==========================================
    st.divider()
    st.caption(
        "Student Course Completion Predictor | "
        "Machine Learning Project"
    )

except FileNotFoundError:
    st.error(
        "ไม่พบไฟล์โมเดลหรือ Scaler กรุณาวางไฟล์ "
        "student_completion_model.joblib และ scaler.joblib "
        "ไว้ในโฟลเดอร์เดียวกับ app.py"
    )

except Exception as e:
    st.error(f"เกิดข้อผิดพลาด: {e}")
    st.info(
        "โปรดตรวจสอบว่า Feature ที่ใช้ตอนทำนาย "
        "ตรงกับ Feature ที่ใช้ตอนฝึกโมเดล"
    )
