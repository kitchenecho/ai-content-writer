import streamlit as st
import pandas as pd
import requests
import json
import time
import re
import os
import base64
import markdown
from openai import OpenAI

# ==========================================
# 1. LONGCAT / AI ENGINE (HUMANIZED)
# ==========================================

class LongCatEngine:
    def __init__(self, api_keys, base_url, model_name):
        # Support both single key (string) and multiple keys (list)
        if isinstance(api_keys, str):
            self.api_keys = [api_keys]
        else:
            self.api_keys = api_keys
        
        self.base_url = base_url
        self.model = model_name
        self.current_key_index = 0
        self.failed_keys = set()
        
        # Initialize with first key
        self.client = OpenAI(api_key=self.api_keys[0], base_url=base_url)

    def _switch_to_next_key(self):
        """Switch to the next available API key"""
        self.failed_keys.add(self.current_key_index)
        
        # Find next available key
        for i in range(len(self.api_keys)):
            if i not in self.failed_keys:
                self.current_key_index = i
                self.client = OpenAI(api_key=self.api_keys[i], base_url=self.base_url)
                st.warning(f"🔄 Switched to API Key #{i+1}")
                return True
        
        return False

    def chat(self, system_prompt, user_prompt, max_tokens=4000):
        max_retries = len(self.api_keys)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.85, 
                    max_tokens=max_tokens 
                )
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a quota/rate limit error
                if any(keyword in error_msg for keyword in ['quota', 'rate limit', 'insufficient', '429', 'limit exceeded']):
                    st.warning(f"⚠️ API Key #{self.current_key_index+1} quota exceeded")
                    
                    # Try to switch to next key
                    if self._switch_to_next_key():
                        continue  # Retry with new key
                    else:
                        st.error("❌ All API keys have exceeded their quota!")
                        return None
                else:
                    # Other errors (not quota related)
                    st.error(f"LongCat API Error: {e}")
                    return None
        
        st.error("❌ All API keys failed")
        return None

    def analyze_semantic_seo(self, keyword, niche):
        system = "You are a specialized SEO entity extractor."
        prompt = f"""
        Analyze the keyword '{keyword}' for the niche '{niche}'.
        Return JSON ONLY:
        {{
            "intent": "...", 
            "entities": ["entity1", "entity2", "entity3", "entity4", "entity5"], 
            "lsi_keywords": ["keyword1", "keyword2", "keyword3"] 
        }}
        """
        response = self.chat(system, prompt)
        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except:
            return {"intent": "Informational", "entities": [], "lsi_keywords": []}

    def generate_structural_outline(self, keyword, niche, seo_data, article_length="long"):
        entities = ", ".join(seo_data['entities'][:10])
        lsi = ", ".join(seo_data['lsi_keywords'])
        
        # Define section counts based on article length
        length_config = {
            "short": {"sections": "3-4", "target_words": "800-1200"},
            "medium": {"sections": "5-7", "target_words": "1500-2500"},
            "long": {"sections": "8-12", "target_words": "3000-5000"}
        }
        
        config = length_config.get(article_length, length_config["long"])
        
        system = "You are an expert SEO content architect specializing in EEAT-optimized, semantic-rich outlines."
        prompt = f"""
        Create a {article_length}-form blog post outline for '{keyword}' in the {niche} niche.
        Target: {config['target_words']} words total
        
        **SEMANTIC SEO REQUIREMENTS:**
        - Integrate these entities naturally: {entities}
        - Use LSI keywords: {lsi}
        - Build topic clusters around the main keyword
        - Cover user intent comprehensively
        
        **EEAT PRINCIPLES:**
        - Show hands-on Experience (real scenarios, personal testing, "I tried this...")
        - Demonstrate Expertise (technical depth, nuanced insights)
        - Build Authority (reference standards like OSHA/ANSI if relevant, brand comparisons)
        - Establish Trust (honest pros/cons, real-world limitations)
        
        **U.S. AUDIENCE CONTEXT:**
        - Use American brands (Milwaukee, DeWalt, Craftsman, Ryobi, etc.)
        - Reference DIY culture, home workshops, regional climates
        - Include OSHA/ANSI standards where applicable
        - Use familiar measurements (inches, feet, Fahrenheit)
        
        **STRUCTURE RULES:**
        1. NO "Introduction" or "Conclusion" headers
        2. Create {config['sections']} sections with varied length headers (some short, some longer)
        3. Mix header styles: questions, statements, emotional hooks
        4. Add natural context (days, locations, reasons) where it feels authentic
        5. Some headers should feel personal ("The Day I...", "Why I Always...")
        
        **EXAMPLES OF GOOD HEADERS:**
        - "Why I Stopped Using [Common Method] After That Tuesday Incident"
        - "The Real Difference Between [Brand A] and [Brand B] (After 6 Months)"
        - "What Nobody Tells You About [Topic]"
        - "How [Something] Changed My [Routine/Approach] in August 2023"
        
        Return ONLY valid JSON:
        {{ "sections": ["Header 1", "Header 2", ...] }}
        """
        response = self.chat(system, prompt, max_tokens=2000)
        try:
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return data.get("sections", [])
        except:
            return [f"My Experience with {keyword}", f"Why {keyword} Matters", "Common Mistakes", "Advanced Tips"]

    def write_section(self, section_title, keyword, niche, full_context, seo_data, is_intro=False, article_length="long"):
        entities = ", ".join(seo_data['entities'])
        lsi = ", ".join(seo_data['lsi_keywords'])
        prev_context = full_context[-2500:] if full_context else "Start of article."
        
        # Adjust section length based on article length
        length_guidance = {
            "short": "Keep this section concise (100-200 words). Be direct and focused.",
            "medium": "Write a moderate section (200-400 words). Balance detail with brevity.",
            "long": "Write a comprehensive section (400-600 words). Provide thorough coverage."
        }
        
        banned_words = (
            "delve, dive deep, in conclusion, summary, vital role, landscape, tapestry, "
            "realm, unlock, unleash, game-changer, bustling, testament, moreover, "
            "furthermore, it is important to note, needless to say"
        )

        style_guide = f"""
        **STYLE RULES (STRICT):**
        1. **Use First-Person ("I", "Me", "My").** Write like a human blogger sharing personal expertise.
        2. **Conversational Tone:** Use short sentences. Use contractions (don't, can't, I've).
        3. **BANNED WORDS:** Do NOT use these words: {banned_words}.
        4. **No Fluff:** Don't start with "In today's digital world..." or broad generalizations. Jump straight into the value.
        5. **Visuals:** Occasionally ask for an image using exactly this tag format: [Image of description].
        6. **{length_guidance[article_length]}**
        """

        if is_intro:
            intro_length = {
                "short": "Write a brief 2-3 paragraph introduction (150-200 words).",
                "medium": "Write a moderate 3-4 paragraph introduction (250-350 words).",
                "long": "Write a comprehensive 4-5 paragraph introduction (400-500 words)."
            }
            instructions = f"""
            Write a personal Introduction for a blog post about '{keyword}'.
            {intro_length[article_length]}
            {style_guide}
            - Start with a personal hook, story, or a bold statement.
            - State the problem clearly using "I" statements.
            - Do NOT write a header. Just the paragraphs.
            """
        else:
            instructions = f"""
            Write the section: "{section_title}".
            {style_guide}
            - Context so far: ...{prev_context}
            - Integrate entities: {entities}
            - Use LSI keywords: {lsi}
            - Do NOT write the header "{section_title}". Start directly with the text.
            """

        system = f"You are a passionate expert blogger in the {niche} niche. You hate corporate jargon and AI-sounding words."
        
        content = self.chat(system, instructions)
        if content:
            content = re.sub(r'^[#\*]+\s?.*?\n', '', content).strip()
        return content

# ==========================================
# 2. MEDIA & WP UTILS
# ==========================================

def generate_image_pollinations(prompt, save_dir):
    try:
        clean_prompt = requests.utils.quote(prompt[:350])
        url = f"https://image.pollinations.ai/prompt/{clean_prompt}?width=1280&height=720&model=flux-realism&nologo=true"
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            filename = f"img_{int(time.time())}_{hash(prompt[:20])}.jpg"
            path = os.path.join(save_dir, filename)
            with open(path, "wb") as f:
                f.write(response.content)
            return path
    except Exception as e:
        print(f"Image Gen Error: {e}")
    return None

def upload_image_to_wp(path, wp_url, wp_user, wp_pass):
    url = f"{wp_url}/wp-json/wp/v2/media"
    auth = base64.b64encode(f"{wp_user}:{wp_pass}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    try:
        with open(path, 'rb') as img:
            files = {'file': img}
            res = requests.post(url, headers=headers, files=files, timeout=60)
            if res.status_code == 201:
                data = res.json()
                return data['id'], data['source_url']
    except Exception as e:
        print(f"WP Upload Error: {e}")
    return None, None

def process_in_content_images(markdown_text, wp_url, wp_user, wp_pass, niche):
    # CORRECTED REGEX PATTERN
    image_tags = re.findall(r"\[Image of ([^\]]+)\]", markdown_text)
    
    if not image_tags:
        return markdown_text
    
    st.info(f"📸 Generating {len(image_tags)} illustrations...")
    updated_text = markdown_text
    progress_bar = st.progress(0)
    
    for i, prompt in enumerate(image_tags):
        full_prompt = f"Editorial photograph, {prompt}, {niche}, highly detailed"
        img_path = generate_image_pollinations(full_prompt, "output_images")
        
        if img_path and wp_url:
            img_id, img_url = upload_image_to_wp(img_path, wp_url, wp_user, wp_pass)
            if img_url:
                # Create HTML
                html = f'<figure class="wp-block-image"><img src="{img_url}" alt="{prompt}" class="wp-image-{img_id}"/></figure>'
                
                # Replace the tag
                target = f"[Image of {prompt}]"
                updated_text = updated_text.replace(target, html, 1)
        
        progress_bar.progress((i + 1) / len(image_tags))
        time.sleep(1)
        
    return updated_text

def post_to_wordpress(title, html_content, featured_img_id, wp_url, wp_user, wp_pass):
    url = f"{wp_url}/wp-json/wp/v2/posts"
    auth = base64.b64encode(f"{wp_user}:{wp_pass}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    
    data = {
        "title": title,
        "content": html_content,
        "status": "draft",
        "featured_media": featured_img_id
    }
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=60)
        if res.status_code in [200, 201]:
            return res.json()['link']
        else:
            st.error(f"WP Post Error: {res.text}")
    except Exception as e:
        st.error(f"WP Connection Error: {e}")
    return None

# ==========================================
# 3. GUI
# ==========================================

st.set_page_config(page_title="Humanized AI Writer", layout="wide", page_icon="✍️")
st.title("✍️ Human-Style Long-Form Writer")

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Multi-API Key Support
    use_multiple_keys = st.checkbox("Use Multiple API Keys", value=False)
    
    if use_multiple_keys:
        st.info("💡 Enter one API key per line. The system will automatically switch when one runs out of quota.")
        api_keys_text = st.text_area(
            "API Keys (one per line)",
            height=150,
            placeholder="sk-key1...\nsk-key2...\nsk-key3..."
        )
        # Parse multiple keys
        api_keys = [key.strip() for key in api_keys_text.split('\n') if key.strip()]
        if api_keys:
            st.success(f"✅ {len(api_keys)} API key(s) loaded")
    else:
        api_key_single = st.text_input("LongCat API Key", type="password")
        api_keys = [api_key_single] if api_key_single else []
    
    base_url = st.text_input("Base URL", value="https://api.longcat.chat/openai/v1")
    model_name = st.selectbox("Model", ["LongCat-Flash-Chat", "gpt-4o"])
    
    st.header("WordPress")
    wp_url = st.text_input("Site URL")
    wp_user = st.text_input("Username")
    wp_pass = st.text_input("App Password", type="password")

col1, col2 = st.columns([1, 2])
with col1:
    niche = st.text_input("Niche / Persona", "Homeopathy Practitioner")
    
    # ARTICLE LENGTH SELECTOR
    st.subheader("📏 Article Length")
    article_length = st.radio(
        "Choose article length:",
        options=["short", "medium", "long"],
        format_func=lambda x: {
            "short": "Short (800-1200 words)",
            "medium": "Medium (1500-2500 words)",
            "long": "Long (3000-5000 words)"
        }[x],
        index=2  # Default to "long"
    )
    
    input_mode = st.radio("Input", ["Single Keyword", "Bulk CSV"])
    keywords = []
    if input_mode == "Single Keyword":
        k = st.text_input("Keyword")
        if k: keywords = [k]
    else:
        f = st.file_uploader("Upload CSV", type="csv")
        if f:
            # Try multiple encodings to handle different CSV formats
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
            df = None
            
            for encoding in encodings_to_try:
                try:
                    f.seek(0)  # Reset file pointer
                    df = pd.read_csv(f, encoding=encoding)
                    st.info(f"✅ CSV loaded successfully using {encoding} encoding")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                except Exception as e:
                    st.error(f"Error with {encoding}: {str(e)}")
                    continue
            
            if df is None:
                st.error("❌ Could not read CSV file. Please ensure it's a valid CSV format.")
            elif "keyword" in df.columns:
                keywords = df['keyword'].tolist()
                st.success(f"📋 Loaded {len(keywords)} keywords")
            else:
                st.error("❌ CSV must have a 'keyword' column")
    
    run_btn = st.button("Start Human Writer", type="primary")

if run_btn and api_keys and keywords:
    engine = LongCatEngine(api_keys, base_url, model_name)
    
    # Show API key status
    if len(api_keys) > 1:
        st.info(f"🔑 Using {len(api_keys)} API keys with automatic failover")
    
    main_progress = st.progress(0)
    
    for idx, keyword in enumerate(keywords):
        st.divider()
        st.markdown(f"### ✍️ Writing: `{keyword}` ({article_length.upper()})")
        
        # 1. SEO & Outline
        with st.status("🧠 Planning content...", expanded=False) as status:
            seo_data = engine.analyze_semantic_seo(keyword, niche)
            body_sections = engine.generate_structural_outline(keyword, niche, seo_data, article_length)
            status.update(label=f"Plan: {len(body_sections)} sections.", state="complete")
            
        # 2. Writing Content
        full_markdown = "" 
        content_box = st.container()
        
        with content_box:
            st.caption("Drafting Introduction...")
            intro = engine.write_section("Introduction", keyword, niche, "", seo_data, is_intro=True, article_length=article_length)
            full_markdown += f"{intro}\n\n"
        
        prog_text = st.empty()
        for i, section in enumerate(body_sections): 
            prog_text.text(f"Drafting Section {i+1}/{len(body_sections)}: {section}...")
            section_content = engine.write_section(section, keyword, niche, full_markdown, seo_data, is_intro=False, article_length=article_length)
            full_markdown += f"## {section}\n{section_content}\n\n"
            
        prog_text.text("Drafting Conclusion...")
        conclusion = engine.write_section("Conclusion", keyword, niche, full_markdown, seo_data, is_intro=False, article_length=article_length)
        full_markdown += f"## Final Thoughts\n{conclusion}" 
        prog_text.empty()
        
        # 3. Images & Publishing
        if wp_url:
            with st.status("🎨 Visuals & Publishing...", expanded=True) as status:
                final_txt = process_in_content_images(full_markdown, wp_url, wp_user, wp_pass, niche)

                st.write("Creating Featured Image...")
                feat_prompt = f"Editorial photograph, {keyword}, {niche}, authentic"
                feat_path = generate_image_pollinations(feat_prompt, "output_images")
                featured_id = 0
                if feat_path:
                    st.image(feat_path, width=200)
                    featured_id, _ = upload_image_to_wp(feat_path, wp_url, wp_user, wp_pass)
                
                final_html = markdown.markdown(final_txt, extensions=['tables', 'fenced_code', 'nl2br'])
                link = post_to_wordpress(keyword.title(), final_html, featured_id, wp_url, wp_user, wp_pass)
                
                if link: st.success(f"Published: {link}")
                status.update(label="Done!", state="complete")
        else:
             final_txt = full_markdown
        
        # Save Local
        if not os.path.exists("blog_outputs"): os.makedirs("blog_outputs")
        safe_name = re.sub(r'[\\/*?:"<>|]', "", keyword.replace(" ", "_"))
        with open(f"blog_outputs/{safe_name}_{article_length}.md", "w", encoding="utf-8") as f:
            f.write(final_txt)
            
        main_progress.progress((idx + 1) / len(keywords))

    st.balloons()
    st.success("All posts completed!")