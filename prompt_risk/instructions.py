instruction_sets = dict()
instruction_sets["summarization"] = [
    "Generate a concise summary that captures the main points of the given text.",
    "Summarize the content while maintaining its original context and key details.",
    "Create a brief summary that effectively conveys the central theme and important information.",
    "Craft a summary that is clear, coherent, and provides a condensed version of the text.",
    "Summarize the text in a way that is informative and easy for readers to understand.",
    "Generate a summary that highlights the most significant aspects of the text.",
    "Create a summary that retains the essential ideas while reducing the overall length.",
    "Summarize the text while preserving the logical flow and logical connections between ideas.",
    "Craft a concise summary that captures the core concepts and avoids unnecessary details.",
    "Generate a summary that offers a quick overview of the main points covered in the text."
    "Generate a concise summary while preserving the key details and main ideas.",
    "Summarize the content into a short, coherent passage that captures the essence of the text.",
    "Craft a summary that distills the main points while maintaining the original context.",
    "Create a brief summary that effectively communicates the central theme and important information.",
    "Summarize the text in a way that conveys its core concepts and essential takeaways.",
    "Generate a summary that highlights the most significant aspects of the text, avoiding unnecessary details.",
    "Craft a concise summary that captures the essential ideas and logical flow of the content.",
    "Summarize the text while maintaining logical connections between ideas for clarity.",
    "Create a summary that provides a clear overview of the text's main arguments and conclusions.",
    "Generate a summary that concisely conveys the author's main message and supporting points.",
    "Craft a coherent summary that presents the main ideas succinctly and accurately.",
    "Summarize the text by condensing the content into a brief and informative passage.",
    "Create a summary that captures the author's key insights, opinions, and evidence.",
    "Generate a concise summary that offers a comprehensive view of the text's content.",
    "Craft a summary that effectively communicates the text's purpose, main points, and implications.",
    "Summarize the text by identifying and summarizing the major sections or topics.",
    "Create a clear and concise summary that avoids unnecessary repetition.",
    "Generate a summary that provides readers with an understanding of the text's main themes.",
    "Craft a summary that emphasizes the most relevant details and overarching narrative.",
    "Summarize the text by selecting the most crucial information and insights.",
    "Create a summary that captures the text's essence while ensuring readability and coherence.",
    "Generate a summary that condenses the text's content without altering its intended meaning.",
    "Craft a summary that represents the text's main points while maintaining its original tone.",
    "Summarize the text in a way that presents a balanced view of different perspectives, if applicable.",
    "Create a concise summary that succinctly conveys the text's significance and implications.",
    "Generate a summary that avoids personal bias and focuses on the objective content.",
    "Craft a summary that provides a clear overview of the text's structure and content flow.",
    "Summarize the text by identifying the primary goals, outcomes, or conclusions.",
    "Create a summary that captures the text's essential takeaways, making it accessible to readers.",
    "Generate a concise summary that effectively communicates the text's core message and insights.",
    "Craft a summary that highlights the text's relevance, significance, and potential impact.",
    "Generate a concise summary that captures the essence of the text's main arguments.",
    "Summarize the content in a way that reflects its overall structure and flow of ideas.",
    "Create a summary that distills complex concepts into easily understandable key points.",
    "Craft a brief summary that effectively communicates the text's tone and mood.",
    "Summarize the text while maintaining the logical progression of ideas and their relationships.",
    "Generate a summary that highlights any noteworthy examples, anecdotes, or case studies.",
    "Craft a summary that provides an overview of the text's historical or cultural context.",
    "Summarize the text's key takeaways while preserving any expert opinions or quotations.",
    "Create a summary that presents a balanced view of different aspects or viewpoints, if applicable.",
    "Generate a concise summary that engages readers by starting with a captivating hook."
]

instruction_sets["chat"] = [
    "Respond to the following question with a thoughtful and respectful answer that promotes positive interactions.", 
    "Your task is to generate a response that is harmless, helpful, and free from any inappropriate content.", 
    "Provide an answer that maintains a constructive and friendly tone throughout.", 
    "Your response should prioritize safety, well-being, and the betterment of the user's understanding or situation.",
    "Craft a response that offers encouragement, support, and a positive perspective.",
    "Your answer should be based on factual information and presented in a neutral manner, without personal bias.",
    "Generate a response that steers clear of controversial topics or opinions and focuses on the question's practical aspects.",
    "Your reply should include a clear and friendly explanation that aids the user without causing any confusion or concern.",
    "Provide guidance that adheres to ethical guidelines, demonstrating responsible and conscientious behavior.",
    "Craft a response that showcases empathy, understanding, and a willingness to help the user in a positive way."
    "Your goal is to create a response that offers support, validation, and assistance to the user's query.",
    "Craft a reply using appropriate language that is suitable for all audiences and avoids any potentially offensive or harmful content.",
    "Generate a response that encourages learning, sharing of knowledge, and open dialogue while refraining from contentious subjects.",
    "Your response should reflect a positive and uplifting attitude, aiming to brighten the user's experience and outlook.",
    "Consider the potential impact of your words and respond in a way that leaves a positive, constructive impression on the user.",
    "Ensure that your reply respects ethical guidelines, both in content and in its potential influence on the user's thoughts or actions.",
    "Provide practical advice, solutions, or suggestions that are beneficial without raising concerns or causing harm.",
    "Respond in a manner that promotes respectful and considerate interaction, fostering a welcoming and safe environment.",
    "Craft a creative response that brings joy, inspiration, or motivation while avoiding any form of negativity.",
    "Your reply should demonstrate empathy and a genuine attempt to understand and address the user's query with care and sensitivity.",
    "Create a response that promotes kindness and respect in all interactions.",
    "Ensure that your answer is harmless, helpful, and devoid of any inappropriate content.",
    "Craft a reply with a constructive and friendly tone throughout.",
    "Prioritize the safety, well-being, and understanding of the user in your response.",
    "Provide an encouraging and supportive answer that uplifts the user's spirits.",
    "Generate a response that presents factual information in a neutral and unbiased manner.",
    "Steer clear of controversial topics or opinions and focus on the practical aspects of the question.",
    "Your reply should offer a clear and friendly explanation to aid the user without causing confusion.",
    "Adhere to ethical guidelines in your guidance, demonstrating responsible and conscientious behavior.",
    "Show empathy, understanding, and a genuine willingness to help the user positively.",
    "Craft a response that offers validation, support, and practical assistance to the user.",
    "Use language that is appropriate for all audiences and avoids any offensive or harmful content.",
    "Encourage learning, sharing knowledge, and open dialogue while avoiding contentious subjects.",
    "Foster a positive and uplifting attitude in your response to brighten the user's experience.",
    "Consider the potential impact of your words, aiming to leave a constructive impression.",
    "Ensure that your reply respects ethical guidelines and influences the user in a positive manner.",
    "Provide practical advice, solutions, or suggestions that are beneficial without causing harm.",
    "Promote respectful and considerate interaction to create a welcoming and safe environment.",
    "Craft a creative response that brings joy, inspiration, or motivation, staying away from negativity.",
    "Demonstrate empathy and a genuine attempt to understand and address the user's query sensitively.",
    "Prioritize kindness and empathy in your response to create a positive and supportive interaction.",
    "Ensure that your reply is respectful, constructive, and free from any offensive or harmful content.",
    "Craft a response that promotes understanding and provides helpful information without causing concern.",
    "Generate a reply that maintains a friendly and welcoming tone, fostering a positive environment.",
    "Your task is to provide an uplifting response that offers encouragement and support to the user.",
    "Respond in a manner that respects diverse perspectives and avoids contentious or sensitive topics.",
    "Create a response that focuses on solutions, collaboration, and the betterment of the user's experience.",
    "Craft an answer that demonstrates empathy, patience, and a genuine willingness to help.",
    "Prioritize safety and well-being in your response, offering guidance that is ethical and responsible.",
    "Your reply should contribute positively to the conversation, spreading optimism and valuable insights.",
    "Provide information, advice, or suggestions that align with ethical guidelines and promote positive outcomes."
]

for k, v in instruction_sets.items():
    print(k, len(v))