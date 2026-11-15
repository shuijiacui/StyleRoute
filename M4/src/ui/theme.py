import base64
from functools import lru_cache

import streamlit as st

from src.project import project_path


@lru_cache(maxsize=2)
def _font_css(name: str) -> str:
    data = base64.b64encode(project_path(f"resources/fonts/{name}.woff").read_bytes()).decode()
    return (
        f"@font-face{{font-family:{name};src:url(data:font/woff;base64,{data}) "
        "format('woff');font-weight:400;font-display:swap;}"
    )


def apply_theme():
    st.html("<style>" + _font_css("ShanhaiReading") + _font_css("ShanhaiBrush") + "</style>")
    st.html(project_path("src/ui/theme.css"))
    st.html("""
        <header class="letter-topbar">
          <a class="letter-brand" href="#landscape-studio" aria-label="山海来信，回到创作区">
            <span class="brand-seal" aria-hidden="true">山<br>海</span>
            <span>山海来信<small>LETTERS FROM LANDSCAPES</small></span>
          </a>
          <span class="topbar-note">山川湖海，总有一处值得落笔。</span>
          <span class="topbar-edition">风景 · 艺术 · 来信</span>
        </header>
        <section class="letter-intro" id="landscape-studio">
          <div class="intro-copy"><p class="eyebrow">A POSTCARD, A LITTLE PIECE OF THE WORLD</p>
          <h1>把风景，<em>寄给远方。</em></h1>
          <p>挑一帧风景，借一抹画意。让那些未说出口的想念，在纸上相遇。</p></div>
          <div class="intro-postmark" aria-hidden="true"><span>山 海 邮 局</span>
          <b>见字<br>如面</b><small>WITH LOVE</small></div>
        </section>
    """)


def empty_preview():
    illustration = base64.b64encode(project_path("src/ui/mountain.svg").read_bytes()).decode()
    st.html(f"""
        <div class="letter-empty">
          <img src="data:image/svg+xml;base64,{illustration}" alt="" aria-hidden="true">
          <p>风景已在眼前，<br>等一抹画意抵达。</p>
          <small>点击「生成风景」，开始这封来信</small>
        </div>
    """)
