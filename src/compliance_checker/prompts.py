import base64
from typing import List, Dict, Optional
import os


def encode_image(image_path: str) -> str:
    """
    Encode an image to base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def create_prompt_for_preprocessing(base64Image):
    exampleImage = encode_image("data/examples/metadata_ex.png")
    examplePrompt = """
        ##Example##
        User:
            Fill in the blanks based on the given image and respond in JSON format. 
            Ensure that any English characters in the drawing code are placed before the numbers, following this format: XX-000.
                圖樣內容(Title)：<drawing_title>
                圖號(DWG. NO.)：<drawing_number>
        GPT:
            {
                "DrawingTitle": '全套管基樁配筋詳圖',
                "DrawingNumber": 'S-201'
            }   
    """
    prompt = """
        Fill in the blanks based on the given image and respond in JSON format. 
        Ensure that any English characters in the drawing code are placed before the numbers, following this format: XX-000.
            圖樣內容(Title)：<drawing_title>
            圖號(DWG. NO.)：<drawing_number>
    """
    systemContentPrompt = ""
    userContent = [
        {"type": "text", "text": examplePrompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{exampleImage}",
                "detail": "high",
            },
        },
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        }
    ]
    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_rerank(checkItem: str, base64Image: str) -> Dict[str, str]:
    """
    Generates a prompt for the GPT-4o model to perform reranking.

    Parameters:
    - checkItem (str): The information of the input inspection form, described in text format.
    - base64Image (str): The base64 encoded image to be scored.

    Returns:
    - Dict[str, str]: A dictionary containing the 'system' and 'user' prompts, formatted for the GPT-4o model.
    """
    
    ex1Prompt = """
        ###Example###
        You MUST adhere to the format provided in the example, and produce your output in an identical format!
        Example 1. **With required information:**
        {
            "CheckItem": '★樁底高程',
            "CheckStandard": 'EL = ___m',
            "ScoringReasoning": 'The detailed drawing includes the necessary component of the pile's base elevation, which is crucial for the "★樁底高程" check item. It specifies the exact value of "-31.40m" for the pile's base elevation, aligning with the verification standard required. The drawing's precise alignment with the check item and standard regarding the pile's base elevation allows for an accurate compliance assessment. Overall, the drawing effectively supports the verification of the "★樁底高程" by providing the exact value for the "Pile bottom elevation," meeting all the criteria for the evaluation.',
            "FinalScore": 10
        }
        Example 2. **With information that can verify the given check standards:**
        {
            "CheckItem": '★樁底高程',
            "CheckStandard": 'EL = -42.40m',
            "ScoringReasoning": 'The detailed drawing provides the essential element for the "Foundation pile elevation" inspection, depicting the pile's base elevation. It indicates a precise measurement of "-31.40m" for the pile's base elevation, which deviates from the given standard "EL = -42.40m". However, this value will act as a reference for subsequent compliance assessments. In summary, the drawing substantially facilitates the confirmation of the "★樁底高程" check item by furnishing the accurate measurement for the "Pile bottom elevation."'
            "FinalScore": 10
        }
    """
    ex2Prompt = """
        ###Example###
        You MUST adhere to the format provided in the example, and produce your output in an identical format!
        Example 3. **With only partial required information:**
        {
            "CheckItem": '★鋼筋籠加工組合-主筋直徑、長度',
            "CheckStandard": '∮___mm、L___m',
            "ScoringReasoning": 'The drawing includes the necessary component by indicating the splice length of the main reinforcement. However, it falls short in providing detailed dimensions and lengths of the main reinforcement. Consequently, the drawing only partially fulfills the evaluation criteria, lacking in areas critical for a comprehensive assessment.'
            "FinalScore": 1
        }
        Example 4. **Missing required information:**
        {
            "CheckItem": '★樁底高程',
            "CheckStandard": 'EL = ___m',
            "ScoringReasoning": 'The detailed drawing, while identifying the splice length of the main reinforcement, fails to address the relevant check item, which is the elevation of the foundation pile. It lacks the necessary components, keywords, or values pertinent to the specified check item and standard. Consequently, the drawing is not suitable for a compliance assessment regarding the elevation of the foundation pile, leading to an overall evaluation that does not meet the required criteria.'
            "FinalScore": 0
        }
        Example 5. **With information that can verify the given check standards:**
        {
            "CheckItem": '★鋼筋籠加工組合-搭接長度',
            "CheckStandard": '依設計圖主筋搭接長度≧140cm',
            "ScoringReasoning": 'The detailed drawing specifies the required elements by denoting the splice length for the main reinforcing rebar. It sets forth a precise criterion "主筋搭接長度 Ls=165cm" for the splice length of the main reinforcement, which exceeds the given standard "依設計圖主筋搭接長度≧140cm" (meaning that the splice length of the main reinforcement must be at least as long as the length specified in the detailed drawings). This figure will serve as a benchmark for subsequent compliance assessments. To conclude, the detailed drawing significantly aids in verifying the check item "★鋼筋籠加工組合-搭接長度" by providing a definitive standard for the "Splice length of the main reinforcement."'
            "FinalScore": 10
        }
        Example 6. **With information that can verify the given check standards:**
        {
            "CheckItem": '★鋼筋籠加工組合-搭接長度',
            "CheckStandard": '依設計圖主筋搭接長度≧210cm',
            "ScoringReasoning": 'The detailed drawing specifies the required elements by denoting the splice length for the main reinforcing rebar, establishing a precise criterion of "主筋搭接長度 Ls=165cm." However, this does not align with the stated standard of "依設計圖主筋搭接長度≧210cm." The discrepancy suggests that the provided standard is incorrect and can be adjusted according to the drawing. In conclusion, the detailed drawing plays a critical role in verifying the check item "★鋼筋籠加工組合-搭接長度" by offering a clear standard for the splice length of the main reinforcement.'
            "FinalScore": 10
        }
        
        **The following incorrect example MUST BE AVOIDED.**
        **Incorrect Example 1:**
        {
            "CheckItem": "★基樁樁徑及樁長",
            "CheckStandard": "L=60m;D=1.8m",
            "ScoringReasoning": "The table provides detailed information about the pile diameter and length for various positions. For position LL123, the pile diameter (樁徑 D) is listed as 120 cm, and the pile length (樁長 PL) is 3100 cm, which translates to 31 meters. This does not match the check standard of L=60m and D=1.8m. The discrepancy in both the diameter and length indicates that the drawing does not meet the specified check standard. The drawing provides accurate data for the LL123 position, but it does not align with the required specifications, suggesting that the check standard may need to be revised based on the actual design requirements.",
            "FinalScore": 2
        }
        **Issues with this Response:**
        1. Misinterpretation of Standard vs. Actual Design Requirements: 
            The response incorrectly penalizes the drawing for not matching the check standard, despite the drawing accurately reflecting the design data. 
            This is a misunderstanding, as the accurate information should prompt an adjustment to the standard, rather than a low score for the drawing.
        2. Lack of Contextual Judgment: 
            Instead of recognizing the drawing's accurate information as a potential tool for revising the check standard, the response implies the drawing is at fault, leading to an unfair score.
        **Instructions for Improvement:**
        1. When the drawing provides accurate and verified information that does not align with the check standard, consider whether the standard itself might need review.
        2. Avoid penalizing the drawing for discrepancies if it offers reliable, detailed data that could help refine the standard.
        3. Assign **higher scores** when the drawing serves as a trustworthy source of information, even if it diverges from the check standard, especially if the standard appears outdated or incorrect.
        **Correct Revision**
        {
            "CheckItem": "★基樁樁徑及樁長",
            "CheckStandard": "L=60m;D=1.8m",
            "ScoringReasoning": "The table provides accurate and detailed information about the pile diameter and length for various positions. For position LL123, the pile diameter (樁徑 D) is listed as 120 cm, and the pile length (樁長 PL) is 3100 cm, equivalent to 31 meters. While this does not align with the check standard of L=60m and D=1.8m, the drawing reflects the true design requirements. This discrepancy suggests that the check standard may be outdated or inaccurate and should be adjusted to match the actual design data. Since the drawing provides a reliable source for reviewing and updating the check standard, it merits a high score as a reference for refining the standard.",
            "FinalScore": 10
        }
        
        **Incorrect Example 2:**
        {
            "CheckItem": "★柱主筋錨固長度",
            "CheckStandard": "錨固長度 ≥ 60 cm",
            "ScoringReasoning": "The drawing contains a table showing the value of '出筋長度' as 55 cm for the column main bars. This is closely related to anchorage length, so we can consider it equivalent. Even though it falls slightly below the standard, the drawing provides accurate dimensional data that can help revise the check standard. Therefore, the drawing is considered a valid reference, and the final score is 10.",
            "FinalScore": 10
        }
        **Issues with this Response:**
        
        1. Incorrect Term Substitution:
            The check item is 錨固長度 (anchorage length), but the drawing only contains 出筋長度 (rebar projection length). These are different technical terms with distinct meanings.
        2. False Equivalence:
            Treating "出筋長度" as a proxy for "錨固長度" without clear evidence introduces error. It violates the instruction that the exact term must be present to consider it directly relevant.
        3. Improper Full Score Justification:
            A perfect score (10) is incorrectly awarded despite the drawing not mentioning the exact inspection keyword and failing to meet the numeric standard.
        4. Misuse of Standard Revision Logic:
            The justification improperly assumes the drawing helps revise the standard, despite the wrong parameter being referenced.
        **Correct Revision**
        {
            "CheckItem": "★柱主筋錨固長度",
            "CheckStandard": "錨固長度 ≥ 60 cm",
            "ScoringReasoning": "The drawing includes a value labeled '出筋長度' (rebar projection length), which is listed as 55 cm. However, this is not the same as '錨固長度' (anchorage length). While they may appear similar, the two refer to different structural characteristics and cannot be interchanged. Since the drawing does not include the exact check item keyword '錨固長度' nor an equivalent explicitly defined parameter, it cannot be considered a valid reference for this check item.",
            "FinalScore": 0
        }
    """
    prompt = f"""
            ###Question### **Refer to the provided Image.**
            For '{checkItem}', analysize the relation with the given image and provide detailed reasoning for your scoring decisions within 300 words, then do the grading. 
            Note that the dimensions in the detailed drawings and tables are primarily in centimeters unless otherwise specified.
            Should you encounter a value in the check standard that is not specified in the detailed drawing, it is essential to assess whether the information provided in the drawing is adequate to rectify the discrepancy.
            If the drawing contains essential information that enables you to correct the check standard, you should assign a high score to the image, regardless of any discrepancies between the drawing and the check standard.
            """
    systemContentPrompt = """
            ###Instruction###
            As a civil engineer specializing in structural design and assessment, your primary task is to evaluate structural drawings or tables and determine their relation with specific check items. 
            You'll be provided with a structural drawing or table along with a set of check item and standard representing critical aspects of structural integrity and safety. 
            Your task is to evaluate each drawing by assigning a score from 0 to 10, indicating its relevance to the corresponding check item and standard, where a score of 0 signifies no relation and a score of 10 denotes a perfect match.
            **When evaluating a drawing for compliance, ensure that THE CHECK ITEM IS EXPLICITLY MENTIONED AND ACCURATELY REPRESENTED IN THE DRAWING.
            ONLY ASSIGN A HIGH SCORE IF THE DRAWING PROVIDES SPECIFIC INFORMATION DIRECTLY RELATED TO THE CHECK ITEM.
            THE DRAWINGS MAY CONTAIN SIMILAR CHECK ITEMS; ENSURE THAT THE EXACT CHECK ITEM REQUIRED FOR THE COMPLIANCE CHECK IS ACCURATELY IDENTIFIED AND PRESENT IN THE DRAWING.
            AVOID AWARDING HIGH SCORES BASED ON ASSUMPTIONS OR INDIRECT REFERENCES; IF THE DRAWING LACKS CLEAR DATA ON THE CHECK ITEM, IT SHOULD NOT BE CONSIDERED FULLY COMPLIANT.**
            You must carefully observe the details on the picture, including text and icons. 
            Then, organize the information related to the check item and standard and response. 
            Begin by providing detailed reasoning for your scoring decisions, highlighting the key features of the drawings that support or detract from compliance with the check item. 
            Then, format the final score out of 10 as json in the following manner:
            {
                "CheckItem": <checkItem>,
                "CheckStandard": <check_standard>,
                "ScoringReasoning": <reasoning>,
                "FinalScore": <score>" 
            }        
            Note that the dimensions in the detailed drawings and tables are primarily in centimeters unless otherwise specified.
            **Revise your response by carefully examining the provided example and incorrect example.**
            **Ensure that your response aligns with the instructions and the approach illustrated in the example.**
            **Assign full score (10) **only when** the keyword is present **and** the drawing provides all necessary, precise data to serve as a perfect reference for compliance checking.**
            Be attentive to the specific terminology and notations used in structural engineering. For accurate scoring, ensure that terms and notattions are correctly identified and not substituted for one another, as each carries a distinct technical meaning.
            You may only award a high score if the **exact check item keyword** is explicitly mentioned. If the keyword is missing, the score cannot exceed the degree of indirect relevance (max 5).  
            Avoid misunderstanding or conflating one technical term for another.
            Do not award points for inferred or indirect references. If the drawing lacks clear, direct data on the check item, it cannot score above a low-to-medium range.
            """
    userContent = [
        {"type": "text", "text": ex1Prompt},
        {"type": "text", "text": ex2Prompt},
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        },
    ]
    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_compliance_check(
    checkItem: str,
    imageCandidates: List[str],
    reasoning: Optional[str] = None,
    compliance: Optional[str] = None,
    advisedStandard: Optional[str] = None,
    feedback: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generates a prompt for the GPT-4o model (Compliance Checker) to perform compliance check.

    Parameters:
    - checkItem (str): The information of the input inspection form to be evaluated.
    - imageCandidates (List[str]): List of candidate images.
    - reasoning (str): Reasoning from the previous output.
    - compliance (str): Compliance from the previous output.
    - advisedStandard (str): Advised Standard from the previous output.
    - feedback (str): Feedback from the evaluator.

    Returns:
    - Dict[str, str]: A dictionary containing the 'system' and 'user' prompts, formatted for the GPT-4o model.
    """
    ex1Prompt = """
        ###Example###
        You MUST adhere to the format provided in the example, and produce your output in an identical format!
        Example 1.
        {
            "CheckItem": '★樁底高程',
            "CheckStandard": 'EL = -31.40m',
            "Reasoning": 'Upon thorough inspection, the detailed drawing clearly shows that the base of the pile is set at an elevation of -31.40 meters, aligning exactly with the given check standard "EL = -31.40m".',
            "ComplianceWithRegulations": true,
            "AdvisedStandard": 'EL = -31.40m'
        }
    """
    ex2Prompt = """
        ###Example###
        You MUST adhere to the format provided in the example, and produce your output in an identical format!
        Example 2.
        {
            "CheckItem": '★鋼筋籠加工組合-搭接長度',
            "CheckStandard": '依設計圖主筋搭接長度≧140cm',
            "Reasoning": 'The meticulous examination of the detailed drawing indicates that the main reinforcement's splice length must exceed 155cm, which is inconsistent with the specified check standard that states the main reinforcement's splice length should be "at least 140cm according to the design diagram".',
            "ComplianceWithRegulations": false,
            "AdvisedStandard": '依設計圖主筋搭接長度≧155cm'
        }
        Example 3.
        {
            "CheckItem": '★樁底高程',
            "CheckStandard": 'EL = -31.40m',
            "Reasoning": 'The drawing clearly identifies the splice length of the main reinforcement should exceed 165cm. However, it is unrelated to the check item and check standard concerning the elevation of the foundation pile.',
            "ComplianceWithRegulations": 'Not Related',
            "AdvisedStandard": 'None',
        }
    """
    prompt = f"""
        ###Question###  Refer to the provided Image.
        For '{checkItem}' with your last response given: 
            Reasoning: {reasoning}, 
            Compliance with regulations: {compliance}, 
            Advised Standard: {advisedStandard} 
            and the feedback from your evaluator: {feedback}. 
        You are required to perform a thorough analysis of your previous response and the evaluator’s feedback, incorporating the provided image. 
        If you believe the evaluator's assessment is incorrect, provide detailed reasoning to support your stance. 
        Your justification must be clear, logical, and convincing.
        The analysis should adhere strictly to a 300-word limit and involve multiple iterations to refine and elevate your response, ensuring it meets or exceeds the standards set by the evaluator.
        If there is no prior response or feedback, you may skip the aforementioned steps.
        Begin by comprehensively understanding the {checkItem} and the feedback from the evaluator.
        Carefully review the provided image to inform your analysis.
        If you believe the evaluator is wrong, present a well-reasoned argument, backed by evidence from the image and the check item context, explaining why their feedback may not be accurate.
        Ensure that your final response is polished and fulfills the evaluation criteria excellently.
        """
    systemContentPrompt = """
        ###Instruction###
        As a civil engineer specializing in structural design and assessment, your primary responsibility is to review each check item and standard for compliance with the structural drawing or table, which serves as the ground truth.
        For each evaluation task, you will receive:
            1. A Structural Drawing or Table in Mandarin: This is the definitive and reliable source. All dimensions and specifications provided in this drawing or table are to be treated as accurate and authoritative.
            Perform compliance checking strictly and exclusively based on the information provided.
            **Please note that, in Mandarin construction drawings, dimensions are expressed in centimeters (cm) by default including notations, unless explicitly stated otherwise.**
            Do not make any assumptions, interpretations, or inferences beyond the supplied data under any circumstances.
            Ensure all output units remain consistent in centimeters to maintain precision, accuracy, and traceability.
            2. A Set of Check Item and Standard in Mandarin: These are the evaluation criteria, but they may contain errors and must be verified for accuracy against the drawing or table.
        Your task is to determine whether the provided Check Standard correctly reflects the specifications shown in the structural drawing or table.
        ** Important Rules:**
        1. The structural drawing or table is always correct. If discrepancies exist between the Check Standard and the drawing/table, the Check Standard must be revised accordingly.
        2. Do not make assumptions or interpretations beyond the data explicitly provided.
        3. Dimensions are expressed in centimeters (cm) by default in Mandarin construction drawings, unless stated otherwise.
        4. Ensure unit consistency in all outputs
        5. Your response must follow this structure:
            - Explanation: Provide a clear and thorough explanation of the relevant details in the drawing or table that either support or contradict the original check standard.
            - Evaluation Result: State explicitly whether the original check standard is "Correct," "Incorrect," or "Not Related."
                -- Use "Not Related" only when the drawing contains no relevant information about the check item.
            - (If Incorrect) Revised Standard: Present a corrected version of the standard that aligns with the drawing or table, maintaining the original format and intent as much as possible.
        Your response should be in English and structured as json in the following manner:
        {
            "CheckItem": <checkItem>,
            "CheckStandard": <check_standard>,
            "Reasoning": <reasoning>,
            "ComplianceWithRegulations": <output only true, false or 'Not Related' indicating that the provided image is not associated with check item>,
            "AdvisedStandard": <advisedStandard, Make sure the output strictly follows the original given check standard, including the correct use of units>
        }
        The primary reference for all check standard should be the detailed drawing provided. Any previously given standards may be incorrect and should be disregarded if they do not align with the specifications in the detailed drawing.
        Should you encounter a value in the check standard that is not specified in the detailed drawing, this discrepancy suggests that the value may be incorrect and requires adjustment based on the information contained in the detailed drawing.
        """
    userContent = [
        {"type": "text", "text": ex1Prompt},
        {"type": "text", "text": ex2Prompt},
        {"type": "text", "text": prompt}
    ]
    for image in imageCandidates:
        base64Image = encode_image(image)
        userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        })
    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_compliance_evaluator(
    checkItem: str,
    imageCandidates: List[str],
    reasoning: Optional[str],
    compliance: Optional[str],
    advisedStandard: Optional[str],
) -> Dict[str, str]:
    """
    Generates a prompt for the GPT-4o model (Evaluator) to perform evaluation of compliance checker's output.

    Parameters:
    - checkItem (str): The information of the input inspection form to be evaluated.
    - imageCandidates (List[str]): List of candidate images.
    - reasoning (str): Reasoning from compliance checker.
    - compliance (str): Compliance from compliance checker.
    - advisedStandard (str): Advised Standard from compliance checker.

    Returns:
    - Dict[str, str]: A dictionary containing the 'system' and 'user' prompts, formatted for the GPT-4o model.
    """
    exPrompt = """
        ###Example###
        You MUST adhere to the format provided in the example, and produce your output in an identical format!
        Example 1.
        {
            "CheckItem": '★鋼筋籠加工組合-搭接長度',
            "CheckStandard": '依設計圖主筋搭接長度≧90cm',
            "ReasoningFromEvaluatee": 'The provided image displays a detailed structural drawing of a reinforced steel cage used in pile construction. The drawing includes specific notes and dimensions relevant to the construction of the steel cage. Notably, the second note on the drawing states that the splice length of the main reinforcement bars (主筋接長度 Ls) is 165cm. This dimension is crucial as it pertains directly to the check item concerning the splice length of the main reinforcement in the steel cage assembly. The check standard provided specifies that the splice length should be at least 90cm, according to the design diagram. The drawing clearly shows a splice length of 165cm, which exceeds the minimum requirement of 90cm set by the check standard.',
            "ComplianceWithRegulationsFromEvaluatee": true,
            "AdvisedStandardFromEvaluatee": '依設計圖主筋搭接長度≧90cm',
            "Feedback": 'The diagram clearly describes that the overlapping length of the rebar must be greater than or equal to 155 centimeters. However, the provided check item indicates that the overlapping length of the rebar only needs to be greater than 90 centimeters. This obviously does not comply with the specifications in the diagram. An insufficient overlap length will lead to inadequate strength of the rebar. All inspection standards must be based on the content of the diagram, therefore, the inspection standard should be revised to be greater than 165 centimeters as more appropriate.',
            "Decision": false
        }
        Example 2.
        {
            "CheckItem": '★鋼筋籠加工組合-搭接長度',
            "CheckStandard": '依設計圖主筋搭接長度≧90cm',
            "ReasoningFromEvaluatee": 'The detailed drawing provided shows a section of a reinforced steel cage used in pile construction. The notes on the drawing specify that the splice length of the main reinforcement bars (主筋) should be at least 155cm (標注2: 主筋接長度 Ls = 155cm). This information directly relates to the check item concerning the splice length of the main reinforcement in the steel cage assembly. The check standard provided in the query states that the splice length should be at least 90cm, which is less than the 155cm specified in the drawing. Therefore, the drawing indicates a higher standard than the one provided in the check standard.',
            "ComplianceWithRegulationsFromEvaluatee": false,
            "AdvisedStandardFromEvaluatee": '依設計圖主筋搭接長度≧155cm',
            "Feedback": 'The provided image showcases a comprehensive structural drawing of a reinforced steel cage utilized in construction projects. It features detailed annotations concerning the splice length of the main reinforcement bars. The drawing stipulates that the required splice length (Ls) must be a minimum of 155cm, directly correlating with the check item that addresses the splice length of the main reinforcement within the steel cage structure. It explicitly states that the splice length must not fall below 155cm, aligning with the given check standard. Therefore, the indicated value should be amended from 90cm to 155cm to comply with the specified standard.',
            "Decision": true
        }
    """
    prompt = f"""
        ###Question### Refer to the provided Image.
        For '{checkItem}' with the evaluatee's response containing: 
            Reasoning: {reasoning}, 
            Compliance with regulations: {compliance}, 
            Advised Standard: {advisedStandard}, 
        Conduct a thorough examination of the supplied drawing to assess if the given Reasoning, Compliance with Regulations, and Advised Standard accurately align with the specified check item and standard.
    """
    systemContentPrompt = """
        ###Instruction###
        As a civil engineer and evaluator, your responsibility is to assess whether the evaluatee’s analysis and compliance determination are technically accurate, logically sound, and faithfully based on the drawing or table.
        For each evaluation task, you will receive:
            1. A Structural Drawing or Table in Mandarin: This is the ground truth and must be treated as completely accurate. All assessment must be based on the information in this drawing or table.
            Perform evaluation strictly and exclusively based on the information provided.
            **Please note that, in Mandarin construction drawings, dimensions are expressed in CENTIMETERS (cm) by default including notations, unless explicitly stated otherwise.**
            Do not make any assumptions, interpretations, or inferences beyond the supplied data under any circumstances.
            2. A Set of Check Item and Standard in Mandarin: This is what the drawing or table must be checked against. It may contain errors or inaccuracies.
            3. An Evaluatee’s Response – This will include:
                - Explanation: A detailed analysis comparing the standard to the drawing or table.
                - Evaluation Result: A classification as "Correct," "Incorrect," or "Not Related."
                - (If Incorrect) Revised Standard: A modified version of the original standard.
        Evaluation Rules:
        1. The structural drawing or table is the only accurate source of truth. If any discrepancies exist, the Check Standard must be revised, not the drawing.
        2. Your job is to confirm whether the evaluatee has followed this principle.
        3. Do not allow inferences or assumptions beyond what is explicitly stated.
        4. Dimensions in Mandarin drawings are in centimeters (cm) unless otherwise noted.
        Your Evaluation Must Check:
        1. Does the Explanation accurately reflect the contents of the drawing or table?
        2. Is the Evaluation Result consistent with that analysis?
        3. If the result is “Incorrect,” is the proposed Revised Standard aligned with the drawing and formatted properly?
        Your response should be in English and structured as json in the following manner:
        {
            "CheckItem": <checkItem>,
            "CheckStandard": <check_standard>,
            "ReasoningFromEvaluatee": <reasoning from the evaluatee>,
            "ComplianceWithRegulationsFromEvaluatee": <compliance with regulations from the evaluatee>,
            "AdvisedStandardFromEvaluatee": <advised standard from the evaluatee>,
            "Feedback": <explain the inaccuracies or inadequacies and offer specific recommendations on how to rectify the identified issues, make sure the output strictly follows the original given check standard, including the correct use of units>,
            "Decision": <Boolean value true or false. A response of True indicates agreement with the evaluatee's response, while a response of False indicates disagreement>
        }
        The primary reference for all check standard should be the detailed drawing provided. Any previously given standards may be incorrect and should be disregarded if they do not align with the specifications in the detailed drawing.
        Should you encounter a value in the check standard that is not specified in the detailed drawing, this discrepancy suggests that the value may be incorrect and requires adjustment based on the information contained in the detailed drawing.
    """
    userContent = [
        {"type": "text", "text": exPrompt},
        {"type": "text", "text": prompt}
    ]
    for image in imageCandidates:
        base64Image = encode_image(image)
        userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        })
    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_page_retrieval_position(title: str, form: str, position: str, detailedDrawingDir: str):
    """
    Generates a prompt for the GPT models to verify the check position.

    Parameters:
    - title (str): Title of retrieved page.
    - form (str): Form name.
    - position (str): Check position.
    - detailedDrawingDir (str): Directory of detailed drawings from the retrieved page.
    """
    systemContentPrompt = """
        ### Instruction ###
        As a civil engineer specializing in structural design and assessment, your primary task is to determine if the provided drawings and the title of the drawings contain information relevant to the specified check position.
       
        For each evaluation, you will receive:
        1. A series of detailed drawings.
        2. Inspection form information, including the title of the inspection form and the check position.
        
        Your responsibility is to rigorously assess the drawings to determine whether they contain the information corresponding to the given check position.
        **Your analysis should address the following:**
            1. Thoroughly review the drawings, paying close attention to title, tables, and other relevant textual or graphical details.
            2. Compare the content of the drawings against the specified check position.
            3. If the exact check position is not explicitly mentioned or clearly depicted in the drawings, your evaluation should conclude **false**.
            4. Be cautious of similar but non-matching positions. The **EXACT** check position required for the compliance check must be accurately identified and present within the drawings.
            
        Your response should be structured as JSON in the following format:
        {
            "Position": <Check position>,
            "Reasoning": <Explanation of whether the drawing contains information for the given check position>,
            "Decision": <true or false>
        }
    """
    prompt = f"""
        ###Question### 
        Inspection Form: {form}
        Inspection Position: {position}
        Refer to the provided images titled {title}.
        Based on the form information and the drawings given, determine whether the drawings contain the information of the given check position.
    """
    userContent = [
        {"type": "text", "text": prompt},
    ]
    for image in os.listdir(detailedDrawingDir):
        base64Image = encode_image(os.path.join(detailedDrawingDir, image))
        userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        })

    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}

def create_prompt_for_page_retrieval_detail(title: str, form: str, position: str, checkItemsInfo: str, detailedDrawingDir: str):
    """
    Generates a prompt for the GPT models to verify the check position.

    Parameters:
    - title (str): Title of retrieved page.
    - form (str): Form name.
    - position (str): Check position.
    - checkItemsInfo (str): Check Item + Check Standard
    - detailedDrawingDir (str): Directory of detailed drawings from the retrieved page.
    """
    systemContentPrompt = """
        ### Instruction ###
        As a civil engineer specializing in structural design and assessment, your task is to determine whether a set of provided drawings can serve as the authoritative reference ("check standard") for code‑compliance verification based on accompanying check‑item data and form information.

        Inputs (per evaluation):
        1. One or more detailed engineering drawings
        2. A list of check items with associated standard requirements
        
        Evaluation Goals:
        1. Verify if the drawings contain accurate and complete information to act as the primary reference for each check item, even when the provided standards are incorrect or incomplete.
        2. Identify cases where the drawing’s data should override, correct, or supplement the given standards.
        
        Definitions of a "Suitable Check Standard":
        1. Exact Match: The drawing’s value matches the check item and its standard value exactly.
        2. Correction Match: The drawing provides a reliable value that corrects an erroneous standard.
        3. Supplemental Match: The drawing supplies missing but critical information, thus completing the standard.

        Key Considerations:
        1. Discrepancies do not automatically disqualify the drawings. Instead, determine whether the drawing offers superior or more accurate data.
        2. Favor the drawing as the reference when its information is precise, verifiable, and relevant to the check items.

        Analysis Steps:
        1. Compare each drawing detail against the check‑item list and standard values.
        2. Identify discrepancies and assess whether the drawing’s data is more accurate or complete.
        3. Decide if the drawing should:
            Override an incorrect standard value
            Fill in missing data
            Serve directly as the check standard
        
        Output Format:
        Respond in JSON:
        {
            "Reasoning": <A concise, jargon‑free explanation of how the drawing aligns with or corrects each check item>,
            "Decision": <true|false>
        }
        Decision Criteria:
        true: The drawing is sufficiently detailed and accurate to serve as the check standard (including cases where it corrects or supplements the provided data).
        false: The drawing lacks necessary information or reliability to act as the check standard.
        
        Examples:
        Missing information
        {
            "Reasoning": "No rebar dimensions or splice lengths are provided, so the drawings cannot verify the required check items.",
            "Decision": false
        }
        Sufficient information
        {
            "Reasoning": "Drawings include rebar sizes, spacings, and splice lengths matching the check items, so they are suitable as the check standard.",
            "Decision": true
        }
        Correction case
        {
            "Reasoning": "Rebar diameter in the drawing is 20mm (standard listed 16mm), and detailed dimension notes confirm 20mm is correct. Use the drawing to correct the standard.",
            "Decision": true
        }
        **The following incorrect example MUST BE AVOIDED.**
        **Incorrect Example:**
        {
            "Reasoning": "The table provides detailed information about the pile diameter and length for various positions. For position LL123, the pile diameter (樁徑 D) is listed as 120 cm, and the pile length (樁長 PL) is 3100 cm, which translates to 31 meters. This does not match the check standard of L=60m and D=1.8m. The discrepancy in both the diameter and length indicates that the drawing does not meet the specified check standard. The drawing provides accurate data for the LL123 position, but it does not align with the required specifications, suggesting that the check standard may need to be revised based on the actual design requirements.",
            "Decision": false
        }
        **Issues with this Response:**
        1. Misinterpretation of Standard vs. Actual Design Requirements: 
            The response incorrectly penalizes the drawing for not matching the check standard, despite the drawing accurately reflecting the design data. 
            This is a misunderstanding, as the accurate information should prompt an adjustment to the standard, rather than a low score for the drawing.
        2. Lack of Contextual Judgment: 
            Instead of recognizing the drawing's accurate information as a potential tool for revising the check standard, the response implies the drawing is at fault, leading to an unfair score.
        **Instructions for Improvement:**
        1. When the drawing provides accurate and verified information that does not align with the check standard, consider whether the standard itself might need review.
        2. Avoid penalizing the drawing for discrepancies if it offers reliable, detailed data that could help refine the standard.
        3. Assign **higher scores** when the drawing serves as a trustworthy source of information, even if it diverges from the check standard, especially if the standard appears outdated or incorrect.
        **Correct Revision**
        {
            "Reasoning": "The table provides accurate and detailed information about the pile diameter and length for various positions. For position LL123, the pile diameter (樁徑 D) is listed as 120 cm, and the pile length (樁長 PL) is 3100 cm, equivalent to 31 meters. While this does not align with the check standard of L=60m and D=1.8m, the drawing reflects the true design requirements. This discrepancy suggests that the check standard may be outdated or inaccurate and should be adjusted to match the actual design data. Since the drawing provides a reliable source for reviewing and updating the check standard, it merits a high score as a reference for refining the standard.",
            "Decision": true
        }
    """
    prompt = f"""
        ###Question### 
        Inspection Form: {form}
        Inspection Position: {position}
        Refer to the provided images titled {title}.
        Based on the form information, 'check item information '{checkItemsInfo}' and the drawings given, determine whether the drawings are suitable for performing code compliance checks for the specified check item information.
    """
    userContent = [
        {"type": "text", "text": prompt},
    ]
    for image in os.listdir(detailedDrawingDir):
        base64Image = encode_image(os.path.join(detailedDrawingDir, image))
        userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        })

    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_page_selection_with_whole_page(title: str, formInfo: str, checkItemsInfo: str, pageDir: str):
    systemContentPrompt = """
        ### Instruction ###
        As a civil engineer specializing in structural design and assessment, your primary task is to determine if the provided drawings are suitable to serve as the check standard for code compliance based on the associated check item information, form information and the specified check position.

        For each evaluation, you will receive:
        1. A Construction Drawing
        2. Inspection form information, including the title of the inspection form and the check position.
        3. A Series of Check Items and Standards
        
        Your responsibility is to first rigorously assess the drawings to determine whether they contain the information corresponding to the given check position.
        **Your analysis should address the following:**
            1. Carefully inspect the drawings, focusing on the text within title, tables and other relevant details.
            2. Compare the details in the drawings with the specified check position.
            3. If the drawings do not explicitly mention or address the check position, your response should indicate false.

        Then, rigorously assess the drawings to determine if the drawings can serve as a reliable check standard for the given check items, even if the provided standards are incorrect or incomplete.
        A perfect match is defined as either an exact match of the check item and the value in the check standard, or an exact match of the check item when the value in the check standard is incorrect according to the given drawing. 
        Additionally, a perfect match can occur if the drawing contains critical information, such as a value that can adequately correct the value in the check standard or complete missing information in the check standard.

        **Key Points to Consider:**
        - The drawings must be evaluated as a potential primary reference. If the drawings contain accurate and relevant information that can correct or enhance the check items, it should be considered suitable for use as a check standard.
        - Discrepancies between the drawings and the provided standards should not automatically result in the drawings being deemed unsuitable. Instead, focus on whether the drawings offer a better or more accurate representation of the required details.

        **Your analysis should address the following:**
            1. Compare the drawing details with the check items and standards, identifying any discrepancies.
            2. **Critically evaluate whether these discrepancies indicate that the drawing should be used as the correct reference, potentially overriding the provided standards.**
            3. Determine if the drawing contains sufficient and accurate information, to serve as the check standard, correcting any errors or omissions in the provided check items.

        **Important Note:** If the drawing provides reliable and accurate information that better reflects the structural requirements, your decision should be true, indicating that the drawing is suitable to serve as the check standard.
        Your response should be structured as JSON in the following format:
        {
            "Reasoning": <Begin by a comprehensive explanation outlining your analysis, including specific elements that align with or correct the check items and standards.>,
            "Decision": <true or false>
        }
        Your response should reflect your expertise in structural engineering principles, drawing interpretation, and quality assessment, ensuring the highest standards of safety and performance in structural design.
        
        ### Example Outputs ###
        1. **Missing required information:**
            {
                "Reasoning": "There is no relevant information for the given check position. Furthermore, there is no relevant information for the given check items such as rebar splice length, rebar dimensions, etc. Therefore, these drawings are not suitable to serve as the check standard for code compliance.",
                "Decision": false
            }
        2. **With sufficient information:**
            {
                "Reasoning": "The drawings contain the correct check position information and sufficient details such as rebar splice length, rebar dimensions, and spacings. Therefore, these drawings are suitable to serve as the check standard for code compliance.",
                "Decision": true
            }
        3. **With information that can verify the given check standards even if the values do not match:**
            {
                "Reasoning": "The drawings contain the correct check position information and sufficient details such as rebar splice length, rebar dimensions, and spacings. Although some of the values in the drawings do not match the given standard, it indicates that the values in the drawings are correct and can be used to verify the provided check standard. Therefore, these drawings are suitable to serve as the check standard for code compliance.",
                "Decision": true
            }
    """
    prompt = f"""
        ###Question### 
        Refer to the provided images titled {title}.
        Based on the form information '{formInfo}, 'check item information '{checkItemsInfo}' and the drawings given, determine whether the drawings are suitable for performing code compliance checks for the specified check item information.
    """
    userContent = [
        {"type": "text", "text": prompt},
    ]
    base64Image = encode_image(os.path.join(pageDir))
    userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        }
    )

    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}

def create_prompt_for_page_selection_with_detail_drawings(title: str, formInfo: str, checkItemsInfo: str, detailedDrawingDir: str):
    systemContentPrompt = """
        ### Instruction ###
        As a civil engineer specializing in structural design and assessment, your primary task is to determine if the provided drawings are suitable to serve as the check standard for code compliance based on the associated check item information, form information and the specified check position.

        For each evaluation, you will receive:
        1. A Series of Detailed Drawings
        2. Inspection form information, including the title of the inspection form and the check position.
        3. A Series of Check Items and Standards
        
        Your responsibility is to first rigorously assess the drawings to determine whether they contain the information corresponding to the given check position.
        **Your analysis should address the following:**
            1. Carefully inspect the drawings, focusing on the text within title, tables and other relevant details.
            2. Compare the details in the drawings with the specified check position.
            3. If the drawings do not explicitly mention or address the check position, your response should indicate false.

        Then, rigorously assess the drawings to determine if the drawings can serve as a reliable check standard for the given check items, even if the provided standards are incorrect or incomplete.
        A perfect match is defined as either an exact match of the check item and the value in the check standard, or an exact match of the check item when the value in the check standard is incorrect according to the given drawing. 
        Additionally, a perfect match can occur if the drawing contains critical information, such as a value that can adequately correct the value in the check standard or complete missing information in the check standard.

        **Key Points to Consider:**
        - The drawings must be evaluated as a potential primary reference. If the drawings contain accurate and relevant information that can correct or enhance the check items, it should be considered suitable for use as a check standard.
        - Discrepancies between the drawings and the provided standards should not automatically result in the drawings being deemed unsuitable. Instead, focus on whether the drawings offer a better or more accurate representation of the required details.

        **Your analysis should address the following:**
            1. Compare the drawing details with the check items and standards, identifying any discrepancies.
            2. **Critically evaluate whether these discrepancies indicate that the drawing should be used as the correct reference, potentially overriding the provided standards.**
            3. Determine if the drawing contains sufficient and accurate information, to serve as the check standard, correcting any errors or omissions in the provided check items.

        **Important Note:** If the drawing provides reliable and accurate information that better reflects the structural requirements, your decision should be true, indicating that the drawing is suitable to serve as the check standard.
        Your response should be structured as JSON in the following format:
        {
            "Reasoning": <Begin by a comprehensive explanation outlining your analysis, including specific elements that align with or correct the check items and standards.>,
            "Decision": <true or false>
        }
        Your response should reflect your expertise in structural engineering principles, drawing interpretation, and quality assessment, ensuring the highest standards of safety and performance in structural design.
        
        ### Example Outputs ###
        1. **Missing required information:**
            {
                "Reasoning": "There is no relevant information for the given check position. Furthermore, there is no relevant information for the given check items such as rebar splice length, rebar dimensions, etc. Therefore, these drawings are not suitable to serve as the check standard for code compliance.",
                "Decision": false
            }
        2. **With sufficient information:**
            {
                "Reasoning": "The drawings contain the correct check position information and sufficient details such as rebar splice length, rebar dimensions, and spacings. Therefore, these drawings are suitable to serve as the check standard for code compliance.",
                "Decision": true
            }
        3. **With information that can verify the given check standards even if the values do not match:**
            {
                "Reasoning": "The drawings contain the correct check position information and sufficient details such as rebar splice length, rebar dimensions, and spacings. Although some of the values in the drawings do not match the given standard, it indicates that the values in the drawings are correct and can be used to verify the provided check standard. Therefore, these drawings are suitable to serve as the check standard for code compliance.",
                "Decision": true
            }
    """
    prompt = f"""
        ###Question### 
        Refer to the provided images titled {title}.
        Based on the form information '{formInfo}, 'check item information '{checkItemsInfo}' and the drawings given, determine whether the drawings are suitable for performing code compliance checks for the specified check item information.
    """
    userContent = [
        {"type": "text", "text": prompt},
    ]
    for image in os.listdir(detailedDrawingDir):
        base64Image = encode_image(os.path.join(detailedDrawingDir, image))
        userContent.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64Image}",
                "detail": "high",
            },
        })

    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}


def create_prompt_for_conflict_detection(checkItem, image1, image2):
    systemContentPrompt = """
    ### Instruction ###
    As a civil engineer specializing in structural drawing compliance checking, your task is:
        * Review the given check item and its standard against the two provided images.
        * Determine whether:
            1. Both images are required to fill the blank standard,
            2. Only one image contains sufficient information, or
            3. The two images provide conflicting information and require manual correction.
        * Always treat the images as authoritative sources.
        * Do not infer or assume values beyond what is explicitly shown in the drawings.
        * If a conflict exists, clearly state so and recommend manual correction.
        * Note that the dimensions in the detailed drawings and tables are primarily in centimeters unless otherwise specified.
    **Key Points to Consider:**
    1. Output must be in JSON format only.
    2. Include "ImageUsage" with one of the following values:
        * "Both Required"
        * "Single Image (A)"
        * "Single Image (B)"
        * "Conflict"
    3. Units must remain consistent with the original check standard.
    **Final Output JSON format:**
    {
        "CheckItem": <checkItem>,
        "Reasoning": <reasoning>,
        "ImageUsage": <Both Required | Single Image (A) | Single Image (B) | Conflict>
    }
    
    ### Example Outputs ###
    1. **Both Required:**
    {
        "CheckItem": "★基樁樁徑及樁長",
        "Reasoning": "Image A clearly shows the pile diameter as 120cm, while Image B shows the pile length as 30m. Both drawings are necessary to fully satisfy the check standard.",
        "ImageUsage": "Both Required"
    }
    
    2. **Single Image:**
    {
        "CheckItem": "★鋼筋籠搭接長度",
        "Reasoning": "Image A alone provides complete information, showing the splice length is 155cm. Image B is unrelated to this check item.",
        "ImageUsage": "Single Image (A)"
    }
    
    3. **Conflict:**
    {
        "CheckItem": "★樁底高程",
        "Reasoning": "Image A shows the pile base elevation as -31.5m, while Image B shows -32.0m. The two images provide conflicting information and require manual correction.",
        "ImageUsage": "Conflict"
    }
    """
    prompt = f"""
        ###Question### 
        Refer to the two provided images for the check item: {checkItem}.
        Determine the appropriate "ImageUsage" following the rules and output strictly in the required JSON format.
    """
    userContent = [
        {"type": "text", "text": prompt},
    ]
    userContent.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(image1)}",
            "detail": "high",
        },
    })
    userContent.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{encode_image(image2)}",
            "detail": "high",
        },
    })

    systemContent = [{"type": "text", "text": systemContentPrompt}]
    return {"system": systemContent, "user": userContent}

def create_prompt_for_correctness_eval(target: str, source: str) -> dict:
    """
    Generates a prompt to compare advisedStandard (source) against checkStandardGT (target).

    Parameters:
    - target (str): Ground-truth check standard.
    - source (str): Advised standard produced by the compliance checker.

    Returns:
    - Dict with 'system' and 'user' keys.
    """
    system_content_prompt = """
        ###Instruction###
        You are given two sets of check standards: target and source.
        Your task is to determine if the source matches the target and to provide reasoning for your decision.
        For each example, compare the values of the target and source, and respond in JSON format:
        {
            "reasoning": <your reasoning>,
            "decision": <true or false>
        }

        ###Rules###
        1. Judge by the values only. Wording, ordering, punctuation and phrasing (e.g. "依設計圖", "應", "至少") do not matter.
        2. Convert units before comparing (1m = 100cm, 1cm = 10mm). If a target value has no unit, assume the unit the source uses for the same quantity.
        3. The source is correct if every value in the target appears in the source with the same meaning.
           Extra details in the source (bar labels, notes, additional correct values) do NOT make it wrong,
           as long as they do not contradict a target value.
        4. The source may list several values for the same quantity (e.g. per section, per zone, inner/outer hoops).
           It is correct as long as the target value is one of them.
        5. The source is wrong if any target value is missing or only appears with a different number
           (e.g. target spacing 80cm but the source only gives 8cm).
           Rule 4 only goes one way: if the target itself lists several values (e.g. "L=45/58m"),
           every one of them must appear in the source.

        ###Example 1###
        Target: "L=18m;D=1.2m"
        Source: "L=22m;D=1.5m"
        Expected Output:
            {
                "reasoning": "The standard in source is different from target, meaning that the standards don't match.",
                "decision": false
            }

        ###Example 2###
        Target: "依設計圖主筋搭接長度≧165cm"
        Source: "搭接長度應大於等於165cm"
        Expected Output:
            {
                "reasoning": "The standard in source matches with the target, meaning that the standard is correct.",
                "decision": true
            }

        ###Example 3###
        Target: "每斷面至少6個、間距2.5m"
        Source: "鋼筋籠間隔器每斷面6個，沿樁長間距250cm配置，採用D16間隔器"
        Expected Output:
            {
                "reasoning": "The source contains both target values: 6 per section and a spacing of 250cm (= 2.5m). The spacer type D16 is extra detail that does not contradict the target.",
                "decision": true
            }

        ###Example 4###
        Target: "依設計圖主筋直徑=28mm、長度=85cm"
        Source: "主筋直徑：上段28mm、下段22mm；伸展長度=85cm"
        Expected Output:
            {
                "reasoning": "The source lists 28mm for the upper part and 22mm for the lower part. The target diameter 28mm is one of the listed values, and the length 85cm matches.",
                "decision": true
            }

        ###Example 5###
        Target: "L=35/47m;D=1.8m"
        Source: "L=35m;D=1.8m"
        Expected Output:
            {
                "reasoning": "The target lists two pile lengths, 35m and 47m, but the source only gives 35m. The 47m length is missing.",
                "decision": false
            }

        ###Example 6###
        Target: "依設計圖箍筋直徑=16mm、間距=80cm"
        Source: "箍筋直徑16mm，間距8cm"
        Expected Output:
            {
                "reasoning": "The diameter matches, but the source spacing is 8cm while the target spacing is 80cm.",
                "decision": false
            }
    """
    prompt = f"""
        ###Question###
        Target: {target}, Source: {source}
    """
    system_content = [{"type": "text", "text": system_content_prompt}]
    user_content = [{"type": "text", "text": prompt}]
    return {"system": system_content, "user": user_content}
