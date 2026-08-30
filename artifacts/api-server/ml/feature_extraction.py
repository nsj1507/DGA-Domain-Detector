import re
import math
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


VOWELS = set("aeiou")


# ==========================================
# SHANNON ENTROPY
# ==========================================

def shannon_entropy(domain):
    """
    Calculate Shannon entropy of a domain.
    """

    if not domain:
        return 0.0

    counter = Counter(domain)
    length = len(domain)

    entropy = 0.0

    for count in counter.values():
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


# ==========================================
# MAXIMUM CONSECUTIVE DIGITS
# ==========================================

def max_consecutive_digits(domain):
    """
    Find the longest sequence of consecutive digits.

    Example:
    abc12345xyz -> 5
    """

    sequences = re.findall(r'\d+', domain)

    if not sequences:
        return 0

    return max(len(sequence) for sequence in sequences)


# ==========================================
# MAXIMUM CONSECUTIVE CONSONANTS
# ==========================================

def max_consecutive_consonants(domain):
    """
    Find the longest sequence of consecutive consonants.
    """

    sequences = re.findall(
        r'[bcdfghjklmnpqrstvwxyz]+',
        domain.lower()
    )

    if not sequences:
        return 0

    return max(len(sequence) for sequence in sequences)


# ==========================================
# CHARACTER TRANSITIONS
# ==========================================

def character_transitions(domain):
    """
    Count transitions between letters and non-letters.
    """

    transitions = 0

    for i in range(1, len(domain)):

        current_is_letter = domain[i].isalpha()
        previous_is_letter = domain[i - 1].isalpha()

        if current_is_letter != previous_is_letter:
            transitions += 1

    return transitions


# ==========================================
# HANDCRAFTED FEATURES
# ==========================================

def extract_handcrafted_features(domain):
    """
    Extract 19 handcrafted features from a domain.
    """

    # Convert domain to lowercase and remove spaces
    domain = str(domain).lower().strip()

    # --------------------------------------
    # Basic length
    # --------------------------------------

    domain_length = len(domain)

    # --------------------------------------
    # Digits
    # --------------------------------------

    num_digits = sum(
        c.isdigit()
        for c in domain
    )

    digit_ratio = (
        num_digits / domain_length
        if domain_length > 0 else 0
    )

    # --------------------------------------
    # Letters
    # --------------------------------------

    num_letters = sum(
        c.isalpha()
        for c in domain
    )

    # --------------------------------------
    # Vowels
    # --------------------------------------

    num_vowels = sum(
        c in VOWELS
        for c in domain
        if c.isalpha()
    )

    # --------------------------------------
    # Consonants
    # --------------------------------------

    num_consonants = sum(
        c.isalpha() and c not in VOWELS
        for c in domain
    )

    # --------------------------------------
    # Vowel ratio
    # --------------------------------------

    vowel_ratio = (
        num_vowels / num_letters
        if num_letters > 0 else 0
    )

    # --------------------------------------
    # Consonant ratio
    # --------------------------------------

    consonant_ratio = (
        num_consonants / num_letters
        if num_letters > 0 else 0
    )

    # --------------------------------------
    # Special characters
    # --------------------------------------

    num_special_chars = sum(
        not c.isalnum() and c != '.'
        for c in domain
    )

    special_char_ratio = (
        num_special_chars / domain_length
        if domain_length > 0 else 0
    )

    # --------------------------------------
    # Hyphens and dots
    # --------------------------------------

    hyphen_count = domain.count('-')

    dot_count = domain.count('.')

    # --------------------------------------
    # Subdomains
    # --------------------------------------

    subdomain_count = max(
        dot_count - 1,
        0
    )

    # --------------------------------------
    # Unique characters
    # --------------------------------------

    unique_chars = len(
        set(domain)
    )

    unique_char_ratio = (
        unique_chars / domain_length
        if domain_length > 0 else 0
    )

    # --------------------------------------
    # Shannon entropy
    # --------------------------------------

    entropy = shannon_entropy(domain)

    # --------------------------------------
    # Consecutive patterns
    # --------------------------------------

    max_digits = max_consecutive_digits(
        domain
    )

    max_consonants = max_consecutive_consonants(
        domain
    )

    # --------------------------------------
    # Character transitions
    # --------------------------------------

    transitions = character_transitions(
        domain
    )

    # --------------------------------------
    # Final feature vector
    # --------------------------------------

    return np.array([
        domain_length,
        num_digits,
        digit_ratio,
        num_letters,
        num_vowels,
        num_consonants,
        vowel_ratio,
        consonant_ratio,
        num_special_chars,
        special_char_ratio,
        hyphen_count,
        dot_count,
        subdomain_count,
        unique_chars,
        unique_char_ratio,
        entropy,
        max_digits,
        max_consonants,
        transitions
    ], dtype=float)


# ==========================================
# FEATURE NAMES
# ==========================================

FEATURE_NAMES = [
    "domain_length",
    "num_digits",
    "digit_ratio",
    "num_letters",
    "num_vowels",
    "num_consonants",
    "vowel_ratio",
    "consonant_ratio",
    "num_special_chars",
    "special_char_ratio",
    "hyphen_count",
    "dot_count",
    "subdomain_count",
    "unique_chars",
    "unique_char_ratio",
    "shannon_entropy",
    "max_consecutive_digits",
    "max_consecutive_consonants",
    "character_transitions"
]


# ==========================================
# N-GRAM FEATURE EXTRACTOR
# ==========================================

class NGramFeatureExtractor:

    def __init__(self):

        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=2,
            max_features=10000
        )

    def fit(self, domains):
        """
        Learn character n-grams from training domains.
        """

        domains = [
            str(domain).lower().strip()
            for domain in domains
        ]

        self.vectorizer.fit(domains)

        return self

    def transform(self, domains):
        """
        Convert domains into TF-IDF n-gram features.
        """

        domains = [
            str(domain).lower().strip()
            for domain in domains
        ]

        return self.vectorizer.transform(
            domains
        )

    def save(
        self,
        filename="tfidf_vectorizer.pkl"
    ):
        """
        Save the fitted TF-IDF vectorizer.
        """

        joblib.dump(
            self.vectorizer,
            filename
        )

    def load(
        self,
        filename="tfidf_vectorizer.pkl"
    ):
        """
        Load a previously fitted TF-IDF vectorizer.
        """

        self.vectorizer = joblib.load(
            filename
        )

        return self


# ==========================================
# COMPLETE FEATURE EXTRACTION
# FOR A NEW DOMAIN
# ==========================================

def extract_features_for_prediction(
    domain,
    vectorizer
):
    """
    Extract the complete feature vector
    for a new domain.

    Combines:
    - 19 handcrafted features
    - 10,000 TF-IDF character n-gram features
    """

    # --------------------------------------
    # Handcrafted features
    # --------------------------------------

    handcrafted = extract_handcrafted_features(
        domain
    )

    # Convert to 2D array
    handcrafted = handcrafted.reshape(
        1,
        -1
    )

    # --------------------------------------
    # TF-IDF n-gram features
    # --------------------------------------

    ngrams = vectorizer.transform([
        str(domain).lower().strip()
    ])

    # Convert sparse matrix to NumPy array
    ngrams = ngrams.toarray()

    # --------------------------------------
    # Combine both feature types
    # --------------------------------------

    combined = np.hstack([
        handcrafted,
        ngrams
    ])

    return combined