# DEPENDENCIES
import numpy as np
from typing import Dict
from typing import List
from typing import Optional


class ConfidenceAnalyzer:
    def __init__(self):
        self.thresholds         = {"excellent" : 0.90,
                                   "very_good" : 0.80,
                                   "good"      : 0.70,
                                   "fair"      : 0.55,
                                   "low"       : 0.35,
                                  }

        self.source_quality_map = {"reports"       : {"score" : 0.95, 
                                                      "label" : "High-quality clinical reports", 
                                                      "icon"  : "📋",
                                                     },
                                   "drug_reviews"  : {"score" : 0.75,
                                                      "label" : "Patient-generated drug reviews", 
                                                      "icon"  : "💊",
                                                     },
                                   "qa_pairs"      : {"score" : 0.6, 
                                                      "label" : "Community-sourced Q&A", 
                                                      "icon"  : "❓",
                                                     },
                                  }


    def get_confidence_level(self, similarity_score: float) -> str:
        """
        Map similarity score to qualitative confidence level
        """
        if (similarity_score >= self.thresholds["excellent"]):
            return "Excellent"

        elif (similarity_score >= self.thresholds["very_good"]):
            return "Very Good"

        elif (similarity_score >= self.thresholds["good"]):
            return "Good"

        elif (similarity_score >= self.thresholds["fair"]):
            return "Fair"

        elif (similarity_score >= self.thresholds["low"]):
            return "Low"

        else:
            return "Poor"


    def get_confidence_explanation(self, similarity_score: float) -> str:
        """
        Provide a descriptive explanation for the confidence level
        """
        if (similarity_score >= self.thresholds["excellent"]):
            return "🟢 **Excellent Match**: High semantic and clinical alignment."

        elif (similarity_score >= self.thresholds["very_good"]):
            return "🟢 **Very Good Match**: Reliable context and terminology."
        
        elif (similarity_score >= self.thresholds["good"]):
            return "🟩 **Good Match**: Generally aligned, clinically valid."

        elif (similarity_score >= self.thresholds["fair"]):
            return "🟨 **Fair Match**: Partial alignment, verify relevance."

        elif (similarity_score >= self.thresholds["low"]):
            return "🟧 **Low Match**: Weak clinical relevance, interpret cautiously."
        
        else:
            return "🔴 **Poor Match**: Misaligned or irrelevant content."


    def calculate_detailed_confidence(self, explanation: Dict, document: Dict, source_collection: str) -> Dict[str, float]:
        """
        Generate a breakdown of confidence factors. Scaled between 0.0 and 1.0
        """
        token_impacts   = explanation.get("token_importance", [])
        base_similarity = explanation.get("base_similarity", 0.0)

        # Normalize token importance for SHAP-style token heatmap
        if token_impacts:
            scores      = np.array([t.get("importance", 0.0) for t in token_impacts])
            norm_scores = (scores - np.min(scores)) / (np.ptp(scores) + 1e-6)
        
        else:
            norm_scores = np.array([0.0])

        keyword_match_ratio  = np.mean(norm_scores > 0.6)
        negative_token_ratio = np.mean(norm_scores < 0.2)

        # Domain-scoped relevance using clinical ontologies
        specialty_weight     = 1.0 if document.get("medical_specialty") else 0.6
        icd_presence         = "icd_code" in document
        snomed_presence      = "snomed_terms" in document
        rxnorm_presence      = "rxnorm_code" in document
        mesh_presence        = "mesh_terms" in document

        coding_boost         = 0.0
        
        if (icd_presence):
            coding_boost += 0.1

        if (snomed_presence):
            coding_boost += 0.1

        if (rxnorm_presence):
            coding_boost += 0.05

        if (mesh_presence):
            coding_boost += 0.05

        # Trust from source
        source_score         = self.source_quality_map.get(source_collection, {}).get("score", 0.6)
        
        # Detailed confidence breakdown
        detailed_confidence  = {"Keyword Match"        : round(keyword_match_ratio, 2),
                                "Context Relevance"    : round(base_similarity, 2),
                                "Medical Accuracy"     : round(specialty_weight + coding_boost, 2),
                                "Source Quality"       : round(source_score, 2),
                                "Negative Token Ratio" : round(negative_token_ratio, 2),
                               }
        
        return detailed_confidence


    def generate_confidence_reasons(self, explanation: Dict, document: Dict, source_collection: str) -> List[Dict[str, str]]:
        """
        Return human-readable explanations for confidence scoring
        """
        reasons       = list()
        token_impacts = explanation.get("token_importance", [])

        # SHAP-style: highlight impactful tokens
        strong_terms  = [t["token"] for t in token_impacts if t.get("importance", 0) > 0.6]
        if strong_terms:
            reasons.append({"icon"   : "✅",
                            "text"   : f"High-impact terms: {', '.join(strong_terms[:3])}",
                            "impact" : "positive",
                          })

        
        weak_terms = [t["token"] for t in token_impacts if t.get("importance", 0) < -0.4]
        if weak_terms:
            reasons.append({"icon"   : "⚠️",
                            "text"   : f"Potentially misleading terms: {', '.join(weak_terms[:2])}",
                            "impact" : "negative",
                          })

        # Domain-scoped markers
        if (spec := document.get("medical_specialty")):
            reasons.append({"icon"   : "🏥",
                            "text"   : f"Targeted specialty: {spec}",
                            "impact" : "positive",
                          })

        if ("icd_code" in document):
            reasons.append({"icon"   : "🧾",
                            "text"   : f"ICD-10: {document['icd_code']}",
                            "impact" : "positive",
                          })

        if ("snomed_terms" in document):
            reasons.append({"icon"   : "🧠",
                            "text"   : "SNOMED tags detected",
                            "impact" : "positive",
                          })

        if ("rxnorm_code" in document):
            reasons.append({"icon"   : "💊",
                            "text"   : f"RxNorm code: {document['rxnorm_code']}",
                            "impact" : "positive",
                          })

        if ("mesh_terms" in document):
            reasons.append({"icon"   : "🔬",
                            "text"   : "MeSH terms found",
                            "impact" : "positive",
                          })

        # Source trust factor
        source = self.source_quality_map.get(source_collection, 
                                             {"score" : 0.6,
                                              "label" : "Unverified source",
                                              "icon"  : "❓",
                                             }
                                            )
        reasons.append({"icon"   : source["icon"],
                        "text"   : f"Source type: {source['label']}",
                        "impact" : "positive" if source["score"] > 0.8 else "neutral",
                      })

        return reasons
