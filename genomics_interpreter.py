import csv
import json

from openai import OpenAI
client = OpenAI()

def generate_ai_summary(report):
    """
    Use the OpenAI API to generate a friendly, non-medical summary
    of the person's genetic trait results.
    """

    report_json = json.dumps(report, indent=2)

    system_message = (
        "You are a friendly, supportive genetics educator writing for a teenager or adult "
        "with no formal genetics background. You are given structured genetic trait data in JSON.\n\n"
        "Your job:\n"
        "1. Start with a short 'Big Picture' overview (1–2 short paragraphs) summarizing overall themes.\n"
        "2. Then write a 'Highlights by Category' section. For any categories that appear in the JSON "
        "(e.g., Nutrition, Fitness, Sleep, Neurobehavior, Sensory, Appearance), briefly describe 1–3 key points "
        "in simple language. This should still be in paragraph form, not bullet points.\n"
        "3. End with a 'Remember' section emphasizing that genetics is only one factor and that environment, "
        "lifestyle, mental health, and medical care matter a lot.\n\n"
        "Important rules:\n"
        "- Do NOT give medical advice.\n"
        "- Do NOT diagnose or suggest treatments.\n"
        "- Do NOT mention specific SNP IDs or genotypes; focus on the meaning.\n"
        "- Keep the tone warm, encouraging, and non-alarming.\n"
        "- Write in clear paragraphs, no markdown symbols like ** or bullet points.\n"
    )

    user_message = (
        "Here is the JSON report describing this person's interpreted genetic traits:\n\n"
        f"{report_json}\n\n"
        "Please follow the instructions in the system message and write the summary accordingly."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()

TRAIT_DB_PATH = "trait_database.csv"
TRAIT_DB_JSON_PATH = "trait_database_model.json"
GENOTYPE_FILE_PATH = "test_genotype.txt"


def generate_text_report(report):
    lines = []
    lines.append("AI-READY GENETIC TRAIT SUMMARY")
    lines.append("=" * 40)
    lines.append(f"Number of traits interpreted: {report['summary']['num_traits_found']}")
    lines.append("Categories: " + ", ".join(report["summary"]["categories"]))
    lines.append("")

    # Group traits by category
    traits_by_cat = {}
    for t in report["traits"]:
        traits_by_cat.setdefault(t["category"], []).append(t)

    for category, traits in traits_by_cat.items():
        lines.append(f"\n## {category}")
        lines.append("-" * (4 + len(category)))

        for t in traits:
            lines.append(f"\nTrait: {t['trait_name']}")
            lines.append(f"Gene: {t['gene']} ({t['rsid']}) — Genotype: {t['user_genotype']}")
            lines.append(f"Effect: {t['effect_label']}  [{t['effect_level']}]")
            lines.append(f"Explanation: {t['explanation']}")
            lines.append(f"Evidence: {t['evidence_strength']}")
            lines.append("")

    return "\n".join(lines)


def load_trait_database(csv_path):
    """Load trait definitions into a lookup dict.

    Primary source: JSON model in TRAIT_DB_JSON_PATH (if present).
    Fallback: legacy CSV at `csv_path`.

    Returns:
        dict keyed by (rsid, genotype) -> row dict with fields matching the
        original CSV shape used by `match_traits`.
    """
    lookup = {}

    # First, try JSON model
    try:
        with open(TRAIT_DB_JSON_PATH, encoding="utf-8") as jf:
            data = json.load(jf)

        for idx, tr in enumerate(data):
            rsid = (tr.get("rsid") or "").strip()
            genotype = (tr.get("genotype") or "").strip().upper()
            if not rsid or not genotype:
                continue

            # Map JSON trait to the legacy row structure used elsewhere
            row = {
                "trait_id": tr.get("trait_id") or f"{rsid}_{genotype}_{idx}",
                "trait_name": tr.get("trait_name", ""),
                "category": tr.get("trait_category", ""),
                "rsid": rsid,
                "gene": tr.get("gene", ""),
                "effect_label": tr.get("variant_effect", ""),
                # Use explicit effect_level if present; otherwise reuse variant_effect as a label
                "effect_level": tr.get("effect_level") or tr.get("variant_effect", ""),
                "explanation": tr.get("explanation", ""),
                "evidence_strength": tr.get("evidence_level", ""),
            }
            key = (rsid, genotype)
            lookup[key] = row

        if lookup:
            return lookup
    except FileNotFoundError:
        # No JSON model yet; fall back to CSV
        pass
    except Exception as e:
        print("Failed to load JSON trait database, falling back to CSV:", e)

    # Fallback: legacy CSV loading
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rsid = row["rsid"].strip()
                genotype = row["genotype"].strip().upper()
                key = (rsid, genotype)
                lookup[key] = row
    except Exception as e:
        print("Failed to load CSV trait database:", e)

    return lookup


def parse_genotype_file(path):
    """
    Parse a 23andMe-style file.
    Returns list of dicts: {rsid, genotype, chromosome, position}
    """
    variants = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expect: rsid  chromosome  position  genotype
            if parts[0].lower() == "rsid":
                # header line
                continue
            if len(parts) < 4:
                continue
            rsid, chrom, pos, genotype = parts[0], parts[1], parts[2], parts[3]
            variants.append(
                {
                    "rsid": rsid,
                    "chromosome": chrom,
                    "position": pos,
                    "genotype": genotype.upper(),
                }
            )
    return variants


def match_traits(trait_lookup, variants):
    """
    For each variant in the user file, check if we have a trait row for (rsid, genotype).
    Returns a list of matched trait objects ready for AI.
    """
    matched_traits = []

    for var in variants:
        key = (var["rsid"], var["genotype"])
        if key in trait_lookup:
            row = trait_lookup[key]
            trait_obj = {
                "trait_id": row["trait_id"],
                "trait_name": row["trait_name"],
                "category": row["category"],
                "rsid": row["rsid"],
                "gene": row["gene"],
                "user_genotype": var["genotype"],
                "effect_label": row["effect_label"],
                "effect_level": row["effect_level"],
                "explanation": row["explanation"],
                "evidence_strength": row["evidence_strength"],
            }
            matched_traits.append(trait_obj)

    return matched_traits


def build_report_object(matched_traits):
    """
    Build a JSON-like object summarizing everything.
    This is what you'd send into an AI model prompt.
    """
    report = {
        "summary": {
            "num_traits_found": len(matched_traits),
            "categories": sorted({t["category"] for t in matched_traits}),
        },
        "traits": matched_traits,
    }
    return report

def effect_level_to_percent(effect_level: str) -> int:
    """
    Map effect_level strings to a rough percentage for the visual bar.
    This is just for visualization, not a true quantitative score.
    """
    level = effect_level.upper()

    # Very rough mapping based on keywords
    if "VERY_HIGH" in level or "HIGH" in level and "LOW" not in level:
        return 85
    if "POWER" in level or "HIGH_RESPONSE" in level:
        return 80
    if "LOW" in level and "TOLERANCE" in level:
        return 25
    if "LOWER" in level or "REDUCED" in level:
        return 35
    if "INTERMEDIATE" in level or "MEDIUM" in level or "MIXED" in level:
        return 55
    if "TYPICAL" in level:
        return 50
    if "ENDURANCE" in level:
        return 60
    if "LIGHT" in level:
        return 65
    if "DARK" in level:
        return 45

    # Fallback
    return 50

def generate_html_report(report, ai_summary=None):
    # Simple icon per category
    category_icons = {
        "Nutrition": "🥦",
        "Fitness": "🏃‍♀️",
        "Sleep": "😴",
        "Neurobehavior": "🧠",
        "Sensory": "👁️",
        "Appearance": "🌈",
    }

    html_parts = []

    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Genetic Trait Report</title>
    <style>
        body {
            font-family: -apple-system, system-ui, -webkit-system-font, sans-serif;
            margin: 0;
            padding: 24px 32px 40px;
            background: #f5f7fb;
            color: #111827;
        }
        h1, h2, h3 {
            color: #111827;
            margin-top: 0;
        }
        h1 {
            font-size: 26px;
            margin-bottom: 4px;
        }
        h2 {
            font-size: 20px;
            margin-top: 22px;
        }
        p {
            margin: 4px 0 8px;
        }
        .header-sub {
            font-size: 0.9em;
            color: #4b5563;
        }
        .page {
            max-width: 900px;
            margin: 0 auto;
            padding: 22px 26px 30px;
            background: #ffffff;
            border-radius: 16px;
            box-shadow: 0 12px 25px rgba(15, 23, 42, 0.12);
        }
        .toc {
            margin: 16px 0 18px;
            padding: 10px 12px;
            background: #f3f4ff;
            border-radius: 12px;
            border: 1px solid #e5e7ff;
            font-size: 0.9em;
        }
        .toc-title {
            font-weight: 600;
            margin-bottom: 6px;
            color: #4338ca;
        }
        .toc-links a {
            margin-right: 10px;
            text-decoration: none;
            color: #4338ca;
            font-weight: 500;
            cursor: pointer;
        }
        .toc-links a:hover {
            text-decoration: underline;
        }
        .category {
            margin-top: 26px;
        }
        .category h2 {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .category-icon {
            font-size: 1.3em;
        }
        .trait-grid {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-top: 6px;
        }
        .trait-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 10px 14px 12px;
            background: #f9fafb;
        }
        .trait-title {
            font-weight: 600;
            font-size: 1.0em;
            margin-bottom: 2px;
        }
        .meta {
            font-size: 0.82em;
            color: #6b7280;
        }
        .effect {
            margin-top: 6px;
            font-size: 0.9em;
        }
        .effect-label {
            font-weight: 500;
        }
        .effect-tag {
            display: inline-block;
            font-size: 0.78em;
            padding: 1px 8px;
            border-radius: 999px;
            background: #eef2ff;
            color: #4338ca;
            margin-left: 6px;
        }
        .bar-outer {
            width: 100%;
            background: #e5e7eb;
            border-radius: 999px;
            height: 8px;
            margin-top: 6px;
            overflow: hidden;
        }
        .bar-inner {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #6366f1, #22c55e);
        }
        .explanation {
            margin-top: 8px;
            font-size: 0.9em;
        }
        .evidence {
            font-size: 0.78em;
            color: #6b7280;
            margin-top: 4px;
        }
        .disclaimer {
            font-size: 0.8em;
            color: #6b7280;
            margin-top: 26px;
            border-top: 1px solid #e5e7eb;
            padding-top: 10px;
        }
        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76em;
            color: #6b7280;
            margin-bottom: 4px;
            font-weight: 600;
        }
        .overview-box {
            margin-top: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            font-size: 0.95em;
        }
    </style>
    <script>
      function scrollToSection(id) {
        var el = document.getElementById(id);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    </script>
</head>
<body>
<div class="page">
""")

    html_parts.append("<h1>Genetic Trait Summary</h1>")
    html_parts.append(
        f"<div class='header-sub'>Traits interpreted: "
        f"<strong>{report['summary']['num_traits_found']}</strong> "
        f"&nbsp;·&nbsp; Categories: {', '.join(report['summary']['categories'])}</div>"
    )

    categories = report["summary"]["categories"]
    if categories:
        html_parts.append("<div class='toc'>")
        html_parts.append("<div class='toc-title'>Jump to a section</div>")
        html_parts.append("<div class='toc-links'>")
        for cat in categories:
            icon = category_icons.get(cat, "🧬")
            safe_id = f"cat-{cat.replace(' ', '-')}"
            html_parts.append(
                f"<a onclick=\"scrollToSection('{safe_id}')\">{icon} {cat}</a>"
            )
        html_parts.append("</div></div>")

    if ai_summary:
        ai_html = ai_summary.replace("\n", "<br>")
        html_parts.append("<div class='section-label'>Personalized overview</div>")
        html_parts.append("<div class='overview-box'>")
        html_parts.append(f"{ai_html}")
        html_parts.append("</div>")

    traits_by_cat = {}
    for t in report["traits"]:
        traits_by_cat.setdefault(t["category"], []).append(t)

    for category, traits in traits_by_cat.items():
        safe_id = f"cat-{category.replace(' ', '-')}"
        icon = category_icons.get(category, "🧬")

        html_parts.append(f'<div class="category" id="{safe_id}">')
        html_parts.append(
            f'<h2><span class="category-icon">{icon}</span>{category}</h2>'
        )
        html_parts.append('<div class="trait-grid">')

        for t in traits:
            from math import floor
            # crude mapping: use effect_level length as proxy if you have no numeric score
            percent = 50
            try:
                level = str(t["effect_level"]).upper()
                if "HIGH" in level and "LOW" not in level:
                    percent = 80
                elif "LOW" in level:
                    percent = 30
                elif "MEDIUM" in level or "INTERMEDIATE" in level or "TYPICAL" in level:
                    percent = 55
            except Exception:
                percent = 50

            html_parts.append('<div class="trait-card">')
            html_parts.append(f'<div class="trait-title">{t["trait_name"]}</div>')
            html_parts.append(
                f'<div class="meta">Gene: {t["gene"]} ({t["rsid"]}) · Genotype: {t["user_genotype"]}</div>'
            )
            html_parts.append(
                '<div class="effect">'
                f'<span class="effect-label">Effect:</span> {t["effect_label"]} '
                f'<span class="effect-tag">{t["effect_level"]}</span>'
                '</div>'
            )
            html_parts.append('<div class="bar-outer">')
            html_parts.append(f'<div class="bar-inner" style="width: {percent}%;"></div>')
            html_parts.append('</div>')
            html_parts.append(f'<div class="explanation">{t["explanation"]}</div>')
            html_parts.append(
                f'<div class="evidence"><strong>Evidence level:</strong> {t["evidence_strength"]}</div>'
            )
            html_parts.append("</div>")

        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.append("""
<div class="disclaimer">
    <strong>Important:</strong> This report is for educational and informational purposes only.
    It does not provide medical advice, diagnosis, or treatment. Genetics is one factor among many including
    environment, sleep, stress, and medical history. For any health related questions, talk with a licensed
    healthcare provider or genetic counselor.
</div>
</div>
</body>
</html>
""")

    return "\n".join(html_parts)


def main():
    trait_lookup = load_trait_database(TRAIT_DB_PATH)
    variants = parse_genotype_file(GENOTYPE_FILE_PATH)
    matched_traits = match_traits(trait_lookup, variants)
    report = build_report_object(matched_traits)

    # JSON output
    print("Matched traits:")
    print(json.dumps(report, indent=2))

    # AI summary
    try:
        ai_summary = generate_ai_summary(report)
        print("\n\nAI Summary:\n")
        print(ai_summary)
    except Exception as e:
        print("AI generation failed:", e)
        ai_summary = None

    # Human-readable text report
    text_report = generate_text_report(report)
    print("\n\nHuman-readable report:\n")
    print(text_report)

    # Save text version
    with open("genetic_report.txt", "w", encoding="utf-8") as f:
        f.write(text_report)

    # Save HTML version (with AI overview if available)
    html_report = generate_html_report(report, ai_summary=ai_summary)
    with open("genetic_report.html", "w", encoding="utf-8") as f:
        f.write(html_report)

    print("\nSaved genetic_report.txt and genetic_report.html")


if __name__ == "__main__":
    main()