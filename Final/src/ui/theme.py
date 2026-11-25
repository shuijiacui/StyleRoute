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
          <a class="brand" href="#landscape-studio" aria-label="山海来信，回到页首">
            <span class="brand-seal" aria-hidden="true">山<br>海</span>
            <span>山海来信<small>LETTERS FROM LANDSCAPES</small></span>
          </a>
        </header>
        <section class="studio-title" id="landscape-studio">
          <h1>把风景，<em>寄给远方。</em></h1>
        </section>
    """)
