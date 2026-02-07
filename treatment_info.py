def get_treatment_recommendations(class_idx):
    """
    Returns treatment recommendations based on the predicted class index
    0: ARMD (Age-Related Macular Degeneration)
    1: BRVO (Branch Retinal Vein Occlusion)
    2: CME (Cystoid Macular Edema)
    3: CWS (Cotton Wool Spots)
    4: HTN (Hypertensive Retinopathy)
    """
    treatments = {
        0: {  # ARMD
            "title": "Treatment Recommendations for Age-Related Macular Degeneration (ARMD)",
            "overview": "ARMD is a progressive eye condition affecting the macula. Treatment depends on whether it's dry or wet ARMD:",
            "treatments": [
                "Anti-VEGF injections (ranibizumab, aflibercept, bevacizumab) for wet ARMD",
                "Photodynamic therapy for certain types of wet ARMD",
                "Low vision rehabilitation and aids",
                "Regular monitoring with Amsler grid at home"
            ],
            "medications": [
                "Anti-VEGF drugs: Lucentis, Eylea, Avastin",
                "AREDS2 formula vitamins (vitamin C, E, zinc, copper, lutein, zeaxanthin) for dry ARMD",
                "Statins may be beneficial in some cases"
            ],
            "diet_lifestyle": [
                "Eat dark leafy greens (spinach, kale) rich in lutein and zeaxanthin",
                "Consume fish high in omega-3 fatty acids",
                "Wear UV-protective sunglasses",
                "Quit smoking (major risk factor)",
                "Maintain healthy blood pressure and cholesterol levels",
                "Exercise regularly"
            ],
            "warning": "Regular follow-up with a retina specialist is crucial. Sudden vision changes require immediate attention."
        },
        1: {  # BRVO
            "title": "Treatment Recommendations for Branch Retinal Vein Occlusion (BRVO)",
            "overview": "BRVO occurs when a branch retinal vein becomes blocked, causing macular edema and vision loss:",
            "treatments": [
                "Anti-VEGF injections to reduce macular edema",
                "Corticosteroid injections (triamcinolone, dexamethasone implant)",
                "Laser photocoagulation for persistent edema or neovascularization",
                "Pan-retinal photocoagulation if neovascularization develops"
            ],
            "medications": [
                "Anti-VEGF drugs: Lucentis, Eylea, Avastin",
                "Corticosteroids: Ozurdex, Kenalog",
                "Blood pressure management medications"
            ],
            "diet_lifestyle": [
                "Control underlying conditions (hypertension, diabetes, glaucoma)",
                "Maintain healthy cholesterol levels",
                "Low-sodium diet",
                "Regular exercise",
                "Smoking cessation",
                "Weight management"
            ],
            "warning": "BRVO requires ongoing monitoring for complications like glaucoma and neovascularization."
        },
        2: {  # CME
            "title": "Treatment Recommendations for Cystoid Macular Edema (CME)",
            "overview": "CME involves fluid accumulation in the macula, often secondary to other conditions:",
            "treatments": [
                "Treat the underlying cause (diabetes, vein occlusion, inflammation)",
                "Topical NSAID eye drops for post-surgical CME",
                "Anti-VEGF injections",
                "Corticosteroid injections or implants",
                "Laser therapy in certain cases",
                "Oral carbonic anhydrase inhibitors"
            ],
            "medications": [
                "Topical NSAIDs: Nepafenac, Ketorolac, Bromfenac",
                "Anti-VEGF agents",
                "Corticosteroids: Prednisone, Triamcinolone",
                "Oral acetazolamide"
            ],
            "diet_lifestyle": [
                "Control underlying systemic diseases",
                "Maintain stable blood sugar levels if diabetic",
                "Low-sodium diet to reduce fluid retention",
                "Regular eye examinations",
                "Protect eyes from UV light"
            ],
            "warning": "CME can cause permanent vision loss if left untreated. Prompt treatment of the underlying cause is essential."
        },
        3: {  # CWS
            "title": "Treatment Recommendations for Cotton Wool Spots (CWS)",
            "overview": "Cotton wool spots represent nerve fiber layer infarctions and indicate underlying systemic conditions:",
            "treatments": [
                "Address and control the underlying systemic disease",
                "Regular retinal examinations to monitor changes",
                "Laser treatment if associated with neovascularization",
                "Manage cardiovascular risk factors aggressively"
            ],
            "medications": [
                "Antihypertensive medications",
                "Diabetes management medications",
                "Cholesterol-lowering drugs",
                "Antiplatelet therapy if indicated"
            ],
            "diet_lifestyle": [
                "Strict blood pressure control",
                "Optimal diabetes management",
                "Low-fat, low-sodium diet",
                "Regular cardiovascular exercise",
                "Smoking cessation",
                "Weight management",
                "Limit alcohol consumption"
            ],
            "warning": "Cotton wool spots often indicate serious systemic conditions requiring comprehensive medical evaluation."
        },
        4: {  # HTN
            "title": "Treatment Recommendations for Hypertensive Retinopathy",
            "overview": "Hypertensive retinopathy results from chronic high blood pressure affecting retinal blood vessels:",
            "treatments": [
                "Aggressive blood pressure control",
                "Regular comprehensive eye examinations",
                "Laser treatment for complications like macular edema",
                "Manage other cardiovascular risk factors"
            ],
            "medications": [
                "Antihypertensive medications (ACE inhibitors, ARBs, beta-blockers, calcium channel blockers)",
                "Diuretics if needed",
                "Cholesterol-lowering medications",
                "Antiplatelet therapy as indicated"
            ],
            "diet_lifestyle": [
                "DASH diet (low sodium, high fruits and vegetables)",
                "Regular aerobic exercise (30 minutes most days)",
                "Weight reduction if overweight",
                "Smoking cessation",
                "Limit alcohol to moderate levels",
                "Stress management techniques",
                "Regular blood pressure monitoring at home"
            ],
            "warning": "Hypertensive retinopathy indicates systemic vascular damage. Urgent blood pressure management is crucial to prevent stroke, heart attack, and other complications."
        }
    }
    
    return treatments.get(class_idx, {})