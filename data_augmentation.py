import random
import pandas as pd
import numpy as np
import nlpaug.augmenter.word as naw

REMOVE_WORD_PROB = 0.8
DEUTSCH_LAN = 'de'
RUSSIAN_LAN = 'ru'
ALL_LANGUAGES = [DEUTSCH_LAN, RUSSIAN_LAN]


class DataAugmentation:
    def __init__(self):
        self.add_word_aug = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased', action="insert")

        self.remove_word_aug = naw.RandomWordAug()

        self.replace_word_aug = naw.SynonymAug(aug_src='wordnet')

        self.back_translation_aug_de = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )

        self.back_translation_aug_ru = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-ru',
            to_model_name='facebook/wmt19-ru-en'
        )

    def add_word_contextual(self, overview_text):
        return self.add_word_aug.augment(overview_text)

    def remove_word_randomly(self, overview_text):
        return self.remove_word_aug.augment(overview_text)

    def replace_word_with_synonym(self, overview_text):
        return self.replace_word_aug.augment(overview_text)

    def back_translation(self, overview_text, language):
        if language == DEUTSCH_LAN:
            return self.back_translation_aug_de.augment(overview_text)
        if language == RUSSIAN_LAN:
            return self.back_translation_aug_ru.augment(overview_text)

    def randomly_augment_text(self, text, remove_word_prob, language):
        # Use back translation
        augmented_text = self.back_translation(text, language)
        removed_words = 0

        # Remove words exponentially with given probability
        while np.random.rand() > remove_word_prob and removed_words < 10:
            augmented_text = self.remove_word_randomly(augmented_text)
            removed_words += 1
            # remove_word_prob = remove_word_prob ** 2

        return augmented_text


def augment_data(df, seed=42):
    # Change seed
    random.seed(seed)
    np.random.seed(seed)

    data_aug = DataAugmentation()

    new_aug_df = df.copy()
    new_aug_df['is_augmented'] = False

    for language in ALL_LANGUAGES:
        df_language = df.copy()

        translations = data_aug.randomly_augment_text(
            df_language['overview'].tolist(), REMOVE_WORD_PROB, language
        )

        df_language['overview'] = translations

        df_language['is_augmented'] = True

        new_aug_df = pd.concat([new_aug_df, df_language])

    return new_aug_df
