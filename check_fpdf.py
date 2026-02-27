try:
    from fpdf import FPDF
    print("fpdf_available")
except ImportError:
    print("fpdf_missing")
