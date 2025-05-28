from __future__ import annotations

import os
import json
from datetime import datetime

import faiss
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from difflib import SequenceMatcher

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# CSV 데이터 로딩 
df = pd.read_csv(
    "https://raw.githubusercontent.com/HisTour-capstone04/HisTour-AI/main/HisTour/heritage/korea_heritage.csv"
)
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

# 전역 RAG 리소스 로딩
model = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("heritage_index.index")
with open("heritage_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# 대화 메모리
messages = [{
    "role": "system",
    "content": "너는 박물관이나 유적지를 안내하는 도슨트 역할을 맡은 AI 챗봇이야.\
사용자가 어떤 문화유산에 대해 질문하면,\
1. 최대한 부드럽고 자연스러운 말투로 이야기하듯 부드럽게 설명해줘.\
2. 간단하게 설명해달라고 하면 3-4줄, 자세히 설명해달라고 하면 그 이상으로 설명해줘.\
3. 너무 전문 용어는 피하고, 처음 듣는 사람도 이해할 수 있도록 말해줘.\
4. 먼저 간단하게 핵심만 요약해주고,\
5. 이어서 흥미로운 배경이나 특징 등을 자세히 덧붙여줘.\
6. 문서에 없는 내용은 절대 추측하지 말고 “자료가 없습니다.”라고 솔직하게 말해줘."
}]

MAX_TURNS = 4
last_mentioned_title = None
pending_choices = None

tmp = (0.0, 0.0)
class ChatRequest(BaseModel):
    user_id: str
    question: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# GPS 거리 필터링 함수
def filter_by_distance(df: pd.DataFrame, user_loc: tuple | None, radius_km: float = 10.0) -> pd.DataFrame:
    if not user_loc:
        return df.reset_index(drop=True)
    lat1, lon1 = map(float, user_loc)
    def haversine(row):
        if row["위도"]=="위도":
            return float('inf')
        lon2 = float(row["위도"])
        lat2 = float(row["경도"])
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))
    df_copy = df.copy()
    df_copy["__dist__"] = df_copy.apply(haversine, axis=1)
    nearby = df_copy[df_copy["__dist__"] <= radius_km].drop(columns="__dist__")
    print("근처 유적지", len(nearby))
    if len(nearby)>0:
        return nearby
    else:
        return df

# 위치 전용 질문 감지
def is_location_only_question(q: str) -> bool:
    kws = ["위치만", "소재지만", "장소만", "위치 알려줘", "소재지 알려줘", "어디야", "어디에 있어"
        , "어디", "위치", "위치를 알려줘", "장소를 알려줘", "장소", "주소", "주소를 알려줘"]
    return any(k in q for k in kws)

# 위치만 응답
def get_location_only(df: pd.DataFrame, question: str):
    cands = []
    for idx, row in df.iterrows():
        title = row['문화재명'].strip()
        if title and title in question:
            cands.append((len(title), idx))
    if not cands:
        return None
    _, best = max(cands)
    r = df.loc[best]
    loc = (r.get('상세주소') or r.get('소재지') or '').strip()
    return f"-->> '{r['문화재명']}'의 위치는 {loc} 입니다." if loc else None

def find_best_matching_title(df, question: str):
    clean_q = question.replace(" ", "").replace("\n", "")
    candidates = []
    for idx, row in df.iterrows():
        title = str(row["문화재명"]).strip()
        t_clean = title.replace(" ", "").replace("\n", "")
        if t_clean and (t_clean in clean_q or clean_q in t_clean):
            candidates.append(idx)
    if not candidates:
        for idx, row in df.iterrows():
            title = str(row["문화재명"]).strip()
            for part in title.split():
                if len(part) >= 2 and part in clean_q:
                    candidates.append(idx)
                    break
    if not candidates:
        return None

    scored = []
    for idx in candidates:
        title   = str(df.loc[idx, "문화재명"]).strip()
        t_clean = title.replace(" ", "").replace("\n", "")
        tokens      = title.split()
        match_count = sum(1 for tok in tokens if tok in question)
        unmatched   = len(tokens) - match_count
        sim         = SequenceMatcher(None, t_clean, clean_q).ratio()
        scored.append(( match_count, -unmatched, sim, -len(tokens), idx ))

    scored.sort(reverse=True)
    print("매칭 문화재", len(scored))
    for i in scored:
        print(i, df.loc[i[4]]["문화재명"])
    _, _, best_sim, _, best_idx = scored[0]
    r = df.loc[best_idx]
    if best_sim>0.25:
        return {
            "title":       r["문화재명"],
            "sim": best_sim,
            "description": r.get("상세설명", "").strip(),
            "location":    (r.get("상세주소","") or r.get("소재지","")).strip(),
        }
    else:
        return None

# GPT-direct (프롬프트 엔지니어링)
def ask_gpt_direct(question: str):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":messages[0]["content"]},
                  {"role":"user","content":question}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

def ask_with_rag(
    question: str,
    top_k: int = 3,
    matches: list[dict] | None = None 
):
    # RAG 검색 (벡터 유사도 top_k)
    qv = model.encode([question])
    D, I = index.search(qv, k=top_k)
    docs = [documents[i] for i in I[0] if documents[i].strip()]

    if not docs:
        return ask_gpt_direct(question)
    # match 정보(제목,유사도 등)를 프롬프트에 정리
    match_ctx = ""
    if matches:
        lines = []
        for m in matches:
            # 예: m = {"title":"경주 불국사","sim":0.44}
            title = m.get("title", "unknown")
            sim   = m.get("sim", "")
            lines.append(f"- {title} (sim: {sim})")
        match_ctx = "\n[매칭된 타이틀]\n" + "\n".join(lines)

    # 문서 컨텍스트
    ctx = "\n---\n".join(docs)

    # 프롬프트 생성 시 match_ctx 삽입
    prompt = f"""
너는 박물관이나 유적지를 안내하는 도슨트 역할을 맡은 AI 챗봇이야.
아래 문서를 참고해서 사용자에게 설명해줘. 
문서에 없는 내용은 '자료가 없습니다.'라고 말하고, 부드럽고 친근한 말투로 이야기하듯 말해줘.

{match_ctx}

[문서]
{ctx}

[질문]
{question}
[답변]
"""

    try:
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system",
                       "content": messages[0]["content"]},
                      {"role":"user","content":f"{match_ctx}\n[문서]\n{ctx}\n[질문]\n{question}"}],
            temperature=0.7
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return ask_gpt_direct(question)

# 챗봇 핵심 로직
def ask_heritage_chatbot(question: str, user_id: str, user_gps: tuple) -> tuple[str, str | None]:
    global messages, last_mentioned_title, pending_choices

    # 사용자 메시지 히스토리에 추가
    messages.append({"role": "user", "content": question})

    if len(messages) > MAX_TURNS * 2 + 1:
        messages = [messages[0]] + messages[-(MAX_TURNS * 2):]
    print(len(messages))

    # 지시어 처리
    pronouns = ["것", "곳", "건", "거기", "여기", "걔", "얘", "방금", "아까", "거긴", "여긴"]
    if any(p in question for p in pronouns) and last_mentioned_title:
        for p in pronouns:
            if p in question:
                question = question.replace(p, last_mentioned_title)
    # GPS 기반 필터링
    # 사용자 위치 기반 필터링
    print("근처 유적지 찾기 시작 : ", datetime.now().second, datetime.now().microsecond)
    sub_df = filter_by_distance(df, user_gps)
    print("그 중에서 매칭 시작 : ", datetime.now().second, datetime.now().microsecond)
    # 위치 전용
    if is_location_only_question(question):
        loc = get_location_only(sub_df, question)
        if loc:
            return loc, None

    # 정확 매칭
    match = find_best_matching_title(sub_df, question)
    if match is None:
        print("거리가 너무 멀어서 전체 데이터에서 찾아보겠음~")
        match = find_best_matching_title(df, question)
    print(match)
    if match:
        print("match")

        title_tokens = match["title"].split(" ")
        matching=1
        for token in title_tokens:
            if token not in question:
                print(token, question)
            else:
                print("**", token, question)
                matching=0
        if matching==1:
            return ask_with_rag(question, matches=[match]), None

        last_mentioned_title = match['title']
        # RAG로도 문서 + DB 통합 검색 실행 (토큰 절감 위해 DB 검색 코드 생략)
        print("매칭 후 답변 생성 시작 : ", datetime.now().second, datetime.now().microsecond)
        return ask_with_rag(question, matches=[match]), match['title']
    # 매칭 없음 → GPT-direct + 저장
    print("non match")
    ans = ask_gpt_direct(question)
    print("매칭 못하고 답변 생성 시작 : ", datetime.now().second, datetime.now().microsecond)
    return ans, None

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat")
async def chat(req: ChatRequest):
    # GPS 정보가 둘 다 있을 때만 tuple, 아니면 None
    if req.latitude is not None and req.longitude is not None:
        user_gps = (float(req.latitude), float(req.longitude))
    else:
        user_gps = None

    answer, match_title = ask_heritage_chatbot(
        question=req.question,
        user_id=req.user_id,
        user_gps=user_gps
    )

    messages.append({"role": "assistant", "content": answer})
    print(messages)
    print("답변 생성 : ", datetime.now().second, datetime.now().microsecond)
    return {"answer": answer, "title": match_title} #여기에 타이틀도 넘겨주기

# latitude가 위도, longitude가 경도 -> 근데 데이터는 반대로 되어 있음.... 그래서 억지로 수정
# uvicorn test_main:app --reload
