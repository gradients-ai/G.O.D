

def reward_specific_char_count(completions, **kwargs):
    """Rewards completions that are close to n_chars characters."""
    n_chars = 100
    return [-abs(n_chars - len(completion)) for completion in completions]


def reward_specific_word_count(completions, **kwargs):
    """Rewards completions that are close to n_words words."""
    n_words = 30
    return [-abs(n_words - len(completion.split())) for completion in completions]


def reward_long_completions(completions, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    return [float(len(completion)) for completion in completions]


def reward_short_completions(completions, **kwargs):
    """Reward function that gives higher scores to shorter completions."""
    return [-float(len(completion)) for completion in completions]


def reward_high_unique_words_percentage(completions, **kwargs):
    """Reward function that gives higher scores to completions with more unique words."""
    scores = []
    for completion in completions:
        words = completion.split()
        if not words:
            scores.append(0.0)
        else:
            scores.append(len(set(words))/len(words))
    return scores


def reward_low_unique_words_percentage(completions, **kwargs):
    """Reward function that gives higher scores to completions with less unique words."""
    scores = []
    for completion in completions:
        words = completion.split()
        if not words:
            scores.append(0.0)
        else:
            scores.append(-len(set(words))/len(words))
    return scores


def reward_think_answer_format(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    import re
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    try:
        matches = [re.match(pattern, content) for content in completions]
        return [1.0 if match else 0.0 for match in matches]
    except Exception as e:
        print(f"Error in format_reward_func: {e}")
        return [0.0 for _ in completions]


def reward_reasoning_keywords(completions, **kwargs):
    """Rewards presence of important reasoning keywords."""
    keywords = [
        # Cause and effect
        'because', 'therefore', 'thus', 'hence', 'consequently', 'as a result',
        # Contrast and comparison
        'however', 'nevertheless', 'nonetheless', 'although', 'despite', 'whereas', 'while',
        # Examples and evidence
        'example', 'instance', 'specifically', 'particularly', 'notably',
        # Addition and sequence
        'furthermore', 'moreover', 'additionally', 'first', 'second', 'finally',
        # Logical connections
        'if', 'then', 'unless', 'since', 'given that', 'assuming that',
        # Analysis and evaluation
        'analyze', 'evaluate', 'consider', 'examine', 'assess', 'determine'
    ]
    return [sum(keyword in completion.lower() for keyword in keywords)
            for completion in completions]


def reward_high_difficult_words_percentage(completions, **kwargs):
    """Rewards text with higher difficult words percentage."""
    import textstat
    scores = []
    for comp in completions:
        words = comp.split()
        if not words:
            scores.append(0.0)
        else:
            difficult_words = textstat.difficult_words(comp)
            scores.append(difficult_words/len(words))
    return scores


def reward_low_difficult_words_percentage(completions, **kwargs):
    """Rewards text with lower difficult words percentage."""
    import textstat
    scores = []
    for comp in completions:
        words = comp.split()
        if not words:
            scores.append(0.0)
        else:
            difficult_words = textstat.difficult_words(comp)
            scores.append(-difficult_words/len(words))
    return scores


def reward_long_sentences(completions, **kwargs):
    """Rewards text with longer average sentence length."""
    import textstat
    scores = [textstat.words_per_sentence(comp) for comp in completions]
    return scores


def reward_short_sentences(completions, **kwargs):
    """Rewards text with shorter average sentence length."""
    import textstat
    scores = [textstat.words_per_sentence(comp) for comp in completions]
    return [-s for s in scores]


def reward_long_words(completions, **kwargs):
    """Rewards text with more characters per word."""
    import textstat
    scores = [textstat.avg_character_per_word(comp) for comp in completions]
    return scores


def reward_short_words(completions, **kwargs):
    """Rewards text with fewer characters per word."""
    import textstat
    scores = [textstat.avg_character_per_word(comp) for comp in completions]
    return [-s for s in scores]


def reward_high_syllables_per_word(completions, **kwargs):
    """Rewards text with more syllables per word."""
    import textstat
    scores = [textstat.avg_syllables_per_word(comp) for comp in completions]
    return scores


def reward_low_syllables_per_word(completions, **kwargs):
    """Rewards text with fewer syllables per word."""
    import textstat
    scores = [textstat.avg_syllables_per_word(comp) for comp in completions]
    return [-s for s in scores]


def reward_high_readability(completions, **kwargs):
    """Rewards more readable text using Flesch reading ease score."""
    import textstat
    scores = [textstat.flesch_reading_ease(comp) for comp in completions]
    return scores


def reward_low_readability(completions, **kwargs):
    """Rewards less readable text using Flesch reading ease score."""
    import textstat
    scores = [textstat.flesch_reading_ease(comp) for comp in completions]
    return [-s for s in scores]


def reward_flesch_kincaid_grade(completions, **kwargs):
    """Rewards text matching target grade level via Flesch-Kincaid Grade Level."""
    import textstat
    target_grade = 12
    scores = [textstat.flesch_kincaid_grade(comp) for comp in completions]
    return [1 - min(abs(s - target_grade)/10, 1) for s in scores]


def reward_positive_sentiment(completions, **kwargs):
    """Rewards completions with more positive sentiment."""
    import langcheck
    scores = langcheck.metrics.sentiment(completions)
    return scores.metric_values


def reward_negative_sentiment(completions, **kwargs):
    """Rewards completions with more negative sentiment."""
    import langcheck
    scores = langcheck.metrics.sentiment(completions)
    return [-s for s in scores.metric_values]


def reward_high_fluency(completions, **kwargs):
    """Rewards completions that are fluent."""
    import langcheck
    scores = langcheck.metrics.fluency(completions)
    return scores.metric_values


def reward_low_fluency(completions, **kwargs):
    """Rewards completions that are less fluent."""
    import langcheck
    scores = langcheck.metrics.fluency(completions)
    return [-s for s in scores.metric_values]


def reward_high_toxicity_score(completions, **kwargs):
    """Rewards completions with higher general toxicity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['toxicity']]


def reward_low_toxicity_score(completions, **kwargs):
    """Rewards completions with lower general toxicity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['toxicity']]


def reward_high_severe_toxicity_score(completions, **kwargs):
    """Rewards completions with higher severe toxicity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['severe_toxicity']]


def reward_low_severe_toxicity_score(completions, **kwargs):
    """Rewards completions with lower severe toxicity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['severe_toxicity']]


def reward_high_obscene_score(completions, **kwargs):
    """Rewards completions with higher obscenity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['obscene']]


def reward_low_obscene_score(completions, **kwargs):
    """Rewards completions with lower obscenity scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['obscene']]


def reward_high_threat_score(completions, **kwargs):
    """Rewards completions with higher threat scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['threat']]


def reward_low_threat_score(completions, **kwargs):
    """Rewards completions with lower threat scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['threat']]


def reward_high_insult_score(completions, **kwargs):
    """Rewards completions with higher insult scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['insult']]


def reward_low_insult_score(completions, **kwargs):
    """Rewards completions with lower insult scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['insult']]


def reward_high_identity_attack_score(completions, **kwargs):
    """Rewards completions with higher identity attack scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [score for score in results['identity_attack']]


def reward_low_identity_attack_score(completions, **kwargs):
    """Rewards completions with lower identity attack scores."""
    from detoxify import Detoxify
    model = Detoxify('original')
    results = model.predict(completions)
    return [1 - score for score in results['identity_attack']]


def test_download_all_rewards():
    """Test all reward functions and trigger model downloads."""
    completions = [
        "",
        "The sky is blue and the grass is green.",
        "I love the smell of rain on a hot day. Oh thank you for the rain.",
        "sky is blue.",
        "I love the smell of rain on a hot day.",
    ]

    try:
        print(f"reward_specific_char_count: \n{reward_specific_char_count(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_specific_char_count: {e}")

    try:
        print(f"reward_specific_word_count: \n{reward_specific_word_count(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_specific_word_count: {e}")

    try:
        print(f"reward_long_completions: \n{reward_long_completions(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_long_completions: {e}")

    try:
        print(f"reward_short_completions: \n{reward_short_completions(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_short_completions: {e}")

    try:
        print(f"reward_high_unique_words_percentage: \n{reward_high_unique_words_percentage(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_unique_words_percentage: {e}")

    try:
        print(f"reward_low_unique_words_percentage: \n{reward_low_unique_words_percentage(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_unique_words_percentage: {e}")

    try:
        print(f"reward_think_answer_format: \n{reward_think_answer_format(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_think_answer_format: {e}")

    try:
        print(f"reward_reasoning_keywords: \n{reward_reasoning_keywords(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_reasoning_keywords: {e}")

    try:
        print(f"reward_high_difficult_words_percentage: \n{reward_high_difficult_words_percentage(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_difficult_words_percentage: {e}")

    try:
        print(f"reward_low_difficult_words_percentage: \n{reward_low_difficult_words_percentage(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_difficult_words_percentage: {e}")

    try:
        print(f"reward_long_sentences: \n{reward_long_sentences(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_long_sentences: {e}")

    try:
        print(f"reward_short_sentences: \n{reward_short_sentences(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_short_sentences: {e}")

    try:
        print(f"reward_long_words: \n{reward_long_words(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_long_words: {e}")

    try:
        print(f"reward_short_words: \n{reward_short_words(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_short_words: {e}")

    try:
        print(f"reward_high_syllables_per_word: \n{reward_high_syllables_per_word(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_syllables_per_word: {e}")

    try:
        print(f"reward_low_syllables_per_word: \n{reward_low_syllables_per_word(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_syllables_per_word: {e}")

    try:
        print(f"reward_high_readability: \n{reward_high_readability(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_readability: {e}")

    try:
        print(f"reward_low_readability: \n{reward_low_readability(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_readability: {e}")

    try:
        print(f"reward_flesch_kincaid_grade: \n{reward_flesch_kincaid_grade(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_flesch_kincaid_grade: {e}")

    try:
        print(f"reward_positive_sentiment: \n{reward_positive_sentiment(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_positive_sentiment: {e}")

    try:
        print(f"reward_negative_sentiment: \n{reward_negative_sentiment(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_negative_sentiment: {e}")

    try:
        print(f"reward_high_fluency: \n{reward_high_fluency(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_fluency: {e}")

    try:
        print(f"reward_low_fluency: \n{reward_low_fluency(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_fluency: {e}")

    try:
        print(f"reward_high_toxicity_score: \n{reward_high_toxicity_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_toxicity_score: {e}")

    try:
        print(f"reward_low_toxicity_score: \n{reward_low_toxicity_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_toxicity_score: {e}")

    try:
        print(f"reward_high_severe_toxicity_score: \n{reward_high_severe_toxicity_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_severe_toxicity_score: {e}")

    try:
        print(f"reward_low_severe_toxicity_score: \n{reward_low_severe_toxicity_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_severe_toxicity_score: {e}")

    try:
        print(f"reward_high_obscene_score: \n{reward_high_obscene_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_obscene_score: {e}")

    try:
        print(f"reward_low_obscene_score: \n{reward_low_obscene_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_obscene_score: {e}")

    try:
        print(f"reward_high_threat_score: \n{reward_high_threat_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_threat_score: {e}")

    try:
        print(f"reward_low_threat_score: \n{reward_low_threat_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_threat_score: {e}")

    try:
        print(f"reward_high_insult_score: \n{reward_high_insult_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_insult_score: {e}")

    try:
        print(f"reward_low_insult_score: \n{reward_low_insult_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_insult_score: {e}")

    try:
        print(f"reward_high_identity_attack_score: \n{reward_high_identity_attack_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_high_identity_attack_score: {e}")

    try:
        print(f"reward_low_identity_attack_score: \n{reward_low_identity_attack_score(completions)}")
    except Exception as e:
        print(f"❌ Error in reward_low_identity_attack_score: {e}")


if __name__ == "__main__":
    test_download_all_rewards()
