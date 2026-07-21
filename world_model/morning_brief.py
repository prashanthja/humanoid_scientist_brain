"""
Layer 9: Morning Brief Email
Sends overnight findings to research teams every morning.
Uses Resend API (free tier: 100 emails/day).
"""
import os, sys, json
sys.path.insert(0, '.')
from dotenv import load_dotenv; load_dotenv('.env')

def get_brief_data(conn):
    """Get overnight findings from world model."""
    cur = conn.cursor()

    # Contradicted beliefs
    cur.execute("""
        SELECT belief_text, contradicting_count, supporting_count, domain
        FROM beliefs
        WHERE contradicting_count > 0
        AND supporting_count >= 2
        AND LENGTH(belief_text) > 30
        AND belief_text NOT LIKE '%does not does not%'
        AND concept_name NOT IN ('Module','time','age')
        ORDER BY contradicting_count DESC LIMIT 3
    """)
    threats = cur.fetchall()

    # New established concepts
    cur.execute("""
        SELECT canonical_name, domain, evidence_count
        FROM concept_cells
        WHERE lifecycle_state = 'established'
        ORDER BY id DESC LIMIT 3
    """)
    established = cur.fetchall()

    # Top hypotheses
    cur.execute("""
        SELECT hypothesis_text, confidence, concept_a, concept_c
        FROM hypotheses
        WHERE tested = FALSE
        ORDER BY confidence DESC LIMIT 3
    """)
    hypotheses = cur.fetchall()

    # Causal discoveries
    cur.execute("""
        SELECT source_concept, relation_type, target_concept, confidence
        FROM causal_relations
        WHERE confidence >= 0.85
        ORDER BY confidence DESC LIMIT 3
    """)
    causal = cur.fetchall()

    # Stats
    cur.execute("SELECT COUNT(*) FROM observations")
    obs = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM beliefs WHERE contradicting_count > 0")
    contested = cur.fetchone()[0]

    return {
        'threats': threats,
        'established': established,
        'hypotheses': hypotheses,
        'causal': causal,
        'observations': obs,
        'contested': contested
    }

def build_email_html(company_name, data):
    """Build HTML email for morning brief."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }}
  .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; }}
  .header {{ background: #03050a; color: white; padding: 24px; }}
  .header h1 {{ margin: 0; font-size: 20px; letter-spacing: 2px; }}
  .header p {{ margin: 4px 0 0; color: #4a6080; font-size: 12px; }}
  .section {{ padding: 20px 24px; border-bottom: 1px solid #f0f0f0; }}
  .section h2 {{ margin: 0 0 12px; font-size: 12px; letter-spacing: 1px; color: #999; text-transform: uppercase; }}
  .item {{ padding: 10px 0; border-bottom: 1px solid #f8f8f8; }}
  .item:last-child {{ border-bottom: none; }}
  .item-title {{ font-size: 13px; color: #1a1a1a; margin-bottom: 4px; }}
  .item-meta {{ font-size: 11px; color: #999; }}
  .threat {{ border-left: 3px solid #ef4444; padding-left: 10px; }}
  .opportunity {{ border-left: 3px solid #10b981; padding-left: 10px; }}
  .hypothesis {{ border-left: 3px solid #3b82f6; padding-left: 10px; }}
  .stats {{ background: #f8f8f8; padding: 16px 24px; display: flex; gap: 20px; }}
  .stat {{ text-align: center; }}
  .stat-val {{ font-size: 20px; font-weight: 600; color: #1a1a1a; }}
  .stat-label {{ font-size: 10px; color: #999; text-transform: uppercase; }}
  .footer {{ padding: 16px 24px; text-align: center; font-size: 11px; color: #999; }}
  .btn {{ display: inline-block; background: #1a6cf0; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 12px; margin-top: 8px; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>TATTVA · MORNING BRIEF</h1>
    <p>Overnight findings for {company_name}</p>
  </div>

  <div class="stats">
    <div class="stat">
      <div class="stat-val">{data['observations']:,}</div>
      <div class="stat-label">Observations</div>
    </div>
    <div class="stat">
      <div class="stat-val">{data['contested']}</div>
      <div class="stat-label">Contested beliefs</div>
    </div>
    <div class="stat">
      <div class="stat-val">{len(data['hypotheses'])}</div>
      <div class="stat-label">Open hypotheses</div>
    </div>
  </div>
"""

    if data['threats']:
        html += '<div class="section"><h2>⚠ Threats to Assumptions</h2>'
        for belief, con, sup, domain in data['threats']:
            html += f"""
    <div class="item threat">
      <div class="item-title">{belief[:120]}</div>
      <div class="item-meta">{domain} · {con} contradicting papers · {sup} supporting</div>
    </div>"""
        html += '</div>'

    if data['causal']:
        html += '<div class="section"><h2>🔗 Causal Discoveries</h2>'
        for src, rel, tgt, conf in data['causal']:
            html += f"""
    <div class="item opportunity">
      <div class="item-title">{src} <span style="color:#999">──[{rel}]──▶</span> {tgt}</div>
      <div class="item-meta">confidence {conf:.0%}</div>
    </div>"""
        html += '</div>'

    if data['hypotheses']:
        html += '<div class="section"><h2>💡 Open Hypotheses</h2>'
        for text, conf, a, c in data['hypotheses']:
            html += f"""
    <div class="item hypothesis">
      <div class="item-title">{(text or '')[:120]}</div>
      <div class="item-meta">confidence {conf:.0%} · untested connection: {a[:30]} → {c[:30]}</div>
    </div>"""
        html += '</div>'

    html += f"""
  <div class="section" style="text-align:center">
    <a href="https://tattvaai.org/company" class="btn">Open Tattva OS →</a>
  </div>
  <div class="footer">Tattva AI · tattvaai.org · Unsubscribe</div>
</div>
</body>
</html>"""
    return html

def send_brief(to_email, company_name, conn):
    """Send morning brief email."""
    try:
        import resend
        resend.api_key = os.environ.get('RESEND_API_KEY', '')
        if not resend.api_key:
            print("No RESEND_API_KEY set")
            return False

        data = get_brief_data(conn)
        html = build_email_html(company_name, data)

        params = {
            "from": "Tattva AI <briefs@tattvaai.org>",
            "to": [to_email],
            "subject": f"Tattva Morning Brief — {len(data['threats'])} threats, {len(data['hypotheses'])} hypotheses",
            "html": html
        }
        email = resend.Emails.send(params)
        print(f"✅ Brief sent to {to_email}: {email}")
        return True
    except Exception as e:
        print(f"❌ Email error: {e}")
        return False

def preview_brief(company_name, conn):
    """Preview brief without sending."""
    data = get_brief_data(conn)
    print(f"\n=== MORNING BRIEF FOR {company_name.upper()} ===")
    print(f"Observations: {data['observations']:,} | Contested: {data['contested']}")
    print(f"\nThreats ({len(data['threats'])}):")
    for t in data['threats']:
        print(f"  ⚠ {t[0][:80]} ({t[3]}, {t[1]} contradicting)")
    print(f"\nCausal discoveries ({len(data['causal'])}):")
    for c in data['causal']:
        print(f"  🔗 {c[0][:25]} --[{c[1]}]--> {c[2][:25]} ({c[3]:.0%})")
    print(f"\nHypotheses ({len(data['hypotheses'])}):")
    for h in data['hypotheses']:
        print(f"  💡 {(h[0] or '')[:80]} ({h[1]:.0%})")
    return data

if __name__ == "__main__":
    import psycopg2
    conn = psycopg2.connect(os.environ.get('DATABASE_URL',''))
    preview_brief("Test Company", conn)
    conn.close()
