try:
    import extra_streamlit_components as stx
    print("stx_found")
except ImportError:
    print("stx_missing")
