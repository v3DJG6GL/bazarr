from __future__ import absolute_import
import logging
import time
from datetime import timedelta

from requests import Session

from requests.exceptions import JSONDecodeError
from subliminal_patch.subtitle import Subtitle
from subliminal_patch.providers import Provider
from subliminal import __short_version__
from subliminal.exceptions import ConfigurationError
from subzero.language import Language
from subliminal.video import Episode, Movie

from babelfish.exceptions import LanguageReverseError

import ffmpeg
import functools
from pycountry import languages
import os

# These are all the languages Whisper supports.
# from whisper.tokenizer import LANGUAGES

whisper_languages = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# Create reverse mapping from alpha3 to alpha2 codes
whisper_alpha3_to_alpha2 = {}
for alpha2, name in whisper_languages.items():
    try:
        lang = Language.fromalpha2(alpha2)
        whisper_alpha3_to_alpha2[lang.alpha3] = alpha2
    except LanguageReverseError:
        continue

def whisper_get_language_reverse(alpha3):
    """Get Whisper language code from alpha3 using precomputed mapping"""
    return whisper_alpha3_to_alpha2.get(alpha3.lower(), None)

language_mapping = {
    "gsw": "deu",  # Swiss German -> German (ISO 639-3)
    "und": "eng", # Undefined -> English
}

logger = logging.getLogger(__name__)

def set_log_level(newLevel="INFO"):
    newLevel = newLevel.upper()
    # print(f'WhisperAI log level changing from {logging._levelToName[logger.getEffectiveLevel()]} to {newLevel}')
    logger.setLevel(getattr(logging, newLevel))

# initialize to default above
set_log_level()

# ffmpeg uses the older ISO 639-2 code when extracting audio streams based on language
# if we give it the newer ISO 639-3 code it can't find that audio stream by name because it's different
# for example it wants 'ger' instead of 'deu' for the German language
#                   or 'fre' instead of 'fra' for the French language
def get_ISO_639_2_code(iso639_3_code):
    # find the language using ISO 639-3 code
    language = languages.get(alpha_3=iso639_3_code)
    # get the ISO 639-2 code or use the original input if there isn't a match
    iso639_2_code = language.bibliographic if language and hasattr(language, 'bibliographic') else iso639_3_code
    logger.debug(f"ffmpeg using language code '{iso639_2_code}' (instead of '{iso639_3_code}')")
    return iso639_2_code

@functools.lru_cache(2)
def encode_audio_stream(path, ffmpeg_path, audio_stream_language=None):
    logger.debug("Encoding audio stream to WAV with ffmpeg")

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        inp = ffmpeg.input(path, threads=0)
        if audio_stream_language:
            # There is more than one audio stream, so pick the requested one by name
            # Use the ISO 639-2 code if available
            audio_stream_language = get_ISO_639_2_code(audio_stream_language)
            logger.debug(f"Whisper will use the '{audio_stream_language}' audio stream for {path}")
            # 0 = Pick first stream in case there are multiple language streams of the same language,
            # otherwise ffmpeg will try to combine multiple streams, but our output format doesn't support that.
            # The first stream is probably the correct one, as later streams are usually commentaries
            lang_map = f"0:m:language:{audio_stream_language}"
        else:
            # there is only one stream, so just use that one
            lang_map = ""
        out, _ = (
            inp.output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, af="aresample=async=1")
            .global_args("-map", lang_map)
            .run(cmd=[ffmpeg_path, "-nostdin"], capture_stdout=True, capture_stderr=True)
        )

    except ffmpeg.Error as e:
        logger.warning(f"ffmpeg failed to load audio: {e.stderr.decode()}")
        return None

    logger.debug(f'Finished encoding audio stream in "{os.path.basename(path)}" with no errors')

    return out


def whisper_get_language(code, name):
    # Whisper uses an inconsistent mix of alpha2 and alpha3 language codes
    try:
        return Language.fromalpha2(code)
    except LanguageReverseError:
        return Language.fromname(name)


def whisper_get_language(code, name):
    """Handle 'und' language code explicitly"""
    if code == "und":
        logger.warning("Undefined language code detected")
        return None
    try:
        return Language.fromalpha2(code)
    except LanguageReverseError:
        try:
            return Language.fromname(name)
        except LanguageReverseError:
            logger.error(f"Could not convert Whisper language: {code} ({name})")
            return None

def language_from_alpha3(lang):
    name = Language(lang).name
    return name

class WhisperAISubtitle(Subtitle):
    '''Whisper AI Subtitle.'''
    provider_name = 'whisperai'
    hash_verifiable = False

    def __init__(self, language, video):
        super(WhisperAISubtitle, self).__init__(language)

        self.video = video
        self.task = None
        self.audio_language = None
        self.force_audio_stream = None

    @property
    def id(self):
        # Construct unique id otherwise provider pool will think
        # subtitles are all the same and drop all except the first one
        # This is important for language profiles with more than one language
        return f"{self.video.original_name}_{self.task}_{str(self.language)}"

    def get_matches(self, video):
        matches = set()

        if isinstance(video, Episode):
            matches.update(["series", "season", "episode"])
        elif isinstance(video, Movie):
            matches.update(["title"])

        return matches


class WhisperAIProvider(Provider):
    '''Whisper AI Provider.'''

    languages = set()

    for lan in whisper_languages:
        languages.update({whisper_get_language(lan, whisper_languages[lan])})

    video_types = (Episode, Movie)

    def __init__(self, endpoint=None, response=None, timeout=None, ffmpeg_path=None, pass_video_name=None, loglevel=None, ambiguous_language_codes=None):
        set_log_level(loglevel)
        if not endpoint:
            raise ConfigurationError('Whisper Web Service Endpoint must be provided')

        if not response:
            raise ConfigurationError('Whisper Web Service Connection/response timeout  must be provided')

        if not timeout:
            raise ConfigurationError('Whisper Web Service Transcription/translation timeout must be provided')

        if not ffmpeg_path:
            raise ConfigurationError('ffmpeg path must be provided')

        if pass_video_name is None:
            raise ConfigurationError('Whisper Web Service Pass Video Name option must be provided')

        self.endpoint = endpoint.rstrip("/")
        self.response = int(response)
        self.timeout = int(timeout)
        self.session = None
        self.ffmpeg_path = ffmpeg_path
        self.pass_video_name = pass_video_name

        # Use provided ambiguous language codes directly without fallback
        self.ambiguous_language_codes = ambiguous_language_codes if ambiguous_language_codes is not None else []
        logger.debug(f"Using ambiguous language codes: {self.ambiguous_language_codes}")

    def initialize(self):
        self.session = Session()
        self.session.headers['User-Agent'] = 'Subliminal/%s' % __short_version__

    def terminate(self):
        self.session.close()


    @functools.lru_cache(2048)
    def detect_language(self, path) -> Language:
        out = encode_audio_stream(path, self.ffmpeg_path)
        if out is None:
            logger.info(f'Whisper cannot detect language of {path} - bad audio track')
            return None

        try:
            video_name = path if self.pass_video_name else None
            r = self.session.post(f"{self.endpoint}/detect-language",
                                params={'encode': 'false', 'video_file': {video_name}},
                                files={'audio_file': out},
                                timeout=(self.response, self.timeout))
            results = r.json()
        except (JSONDecodeError, requests.exceptions.JSONDecodeError):
            logger.error('Invalid JSON response in language detection')
            return None

        if not results.get("language_code"):
            logger.info('Whisper returned empty language code')
            return None

        # Explicitly handle 'und' from Whisper results
        if results["language_code"] == "und":
            logger.info('Whisper detected undefined language')
            return None

        logger.debug(f'Whisper detection raw results: {results}')
        return whisper_get_language(results["language_code"], results["detected_language"])

    def query(self, language, video):
        logger.debug(f'Processing language request: {language.alpha3} ({language_from_alpha3(language.alpha3)}) - File: "{os.path.basename(video.original_path)}"')
        if language not in self.languages:
            logger.debug(f'Language {language.alpha3} not supported by Whisper')
            return None

        sub = WhisperAISubtitle(language, video)
        sub.task = "transcribe"

        # Handle undefined/no audio languages
        if not video.audio_languages:
            logger.debug('No audio language tags present, forcing detection')
            detected_lang = self.detect_language(video.original_path)
            if not detected_lang:
                sub.task = "error"
                sub.release_info = "Language detection failed"
                return sub

            # Apply language mapping after detection
            detected_alpha3 = detected_lang.alpha3
            if detected_alpha3 in language_mapping:
                detected_alpha3 = language_mapping[detected_alpha3]
                logger.debug(f'Mapped detected language {detected_lang} -> {detected_alpha3}')

            sub.audio_language = detected_alpha3

            if detected_alpha3 != language.alpha3:
                sub.task = "translate"
        else:
            # Existing audio language processing with mapping
            processed_languages = {}
            for lang in video.audio_languages:
                if lang in language_mapping:
                    logger.debug(f'Mapping audio language tag: {lang} -> {language_mapping[lang]}')
                mapped_lang = language_mapping.get(lang, lang)
                processed_languages[lang] = mapped_lang

            matched = False
            for original_lang, processed_lang in processed_languages.items():
                if language.alpha3 == processed_lang:
                    sub.audio_language = processed_lang
                    if len(video.audio_languages) > 1:
                        sub.force_audio_stream = original_lang
                    matched = True
                    break

            if not matched:
                sub.task = "translate"
                eligible_languages = [language_mapping.get(l, l) for l in video.audio_languages]
                sub.audio_language = eligible_languages[0] if eligible_languages else None

            # Final validation
            if not sub.audio_language:
                sub.task = "error"
                sub.release_info = "No valid audio language determined"
                return sub
            else:
                # Handle case where audio language exists but may need verification
                # Only run language detection if original unmapped audio languages contain ambiguous codes
                original_ambiguous = any(
                    lang in self.ambiguous_language_codes
                    for lang in video.audio_languages
                )

                if original_ambiguous:
                    logger.debug(
                        f'Original unmapped audio language code {video.audio_languages} is on '
                        f'"Ambiguous Languages Codes" list: {self.ambiguous_language_codes} - forcing detection!'
                    )

                    detected_lang = self.detect_language(video.original_path)
                    if detected_lang is None:
                        sub.task = "error"
                        sub.release_info = "bad/missing audio track - cannot transcribe"
                        return sub

                    detected_alpha3 = detected_lang.alpha3
                    # Apply language mapping after detection
                    if detected_alpha3 in language_mapping:
                        detected_alpha3 = language_mapping[detected_alpha3]

                    sub.audio_language = detected_alpha3
                    sub.task = "transcribe" if detected_alpha3 == language.alpha3 else "translate"

                    logger.debug(
                        f'Post-detection mapping: {detected_lang.alpha3} -> {sub.audio_language} '
                        f'(Requested: {language.alpha3})'
                    )
                else:
                    logger.debug(
                        f'Using existing audio language tag: {sub.audio_language} '
                        f'(originally {video.audio_languages}) - skipping detection'
                    )

        if sub.task == "translate":
            if language.alpha3 != "eng":
                logger.debug(f'Cannot translate from {sub.audio_language} ({language_from_alpha3(sub.audio_language)}) -> {language.alpha3} ({language_from_alpha3(language.alpha3)})! - Only translations to English supported! - File: "{os.path.basename(sub.video.original_path)}"')
                return None

        sub.release_info = f'{sub.task} {language_from_alpha3(sub.audio_language)} audio -> {language_from_alpha3(language.alpha3)} SRT'
        logger.debug(f'WhisperAI query: ({video.original_path}): {sub.audio_language} -> {language.alpha3} - Task: {sub.task}')
        return sub

    def list_subtitles(self, video, languages):
        logger.debug(f'Languages requested from WhisperAI: {", ".join(f"{l.alpha3} ({language_from_alpha3(l.alpha3)})" for l in languages)} - File: "{os.path.basename(video.original_path)}"')
        subtitles = [self.query(l, video) for l in languages]
        return [s for s in subtitles if s is not None]

    def download_subtitle(self, subtitle: WhisperAISubtitle):
        # Invoke Whisper through the API. This may take a long time depending on the file.
        # TODO: This loads the entire file into memory, find a good way to stream the file in chunks

        if subtitle.task == "error":
            return

        out = encode_audio_stream(subtitle.video.original_path, self.ffmpeg_path, subtitle.force_audio_stream)
        if not out:
            logger.info(f"WhisperAI cannot process {subtitle.video.original_path} due to missing/bad audio track")
            subtitle.content = None
            return

        logger.debug(f'Audio stream length: {len(out):,} bytes')
        output_language = "eng" if subtitle.task == "translate" else subtitle.audio_language

        # Convert mapped alpha3 to Whisper's alpha2 code
        input_language = whisper_get_language_reverse(subtitle.audio_language)
        if not input_language:
            if output_language == "eng":
                input_language = "en"
                subtitle.task = "transcribe"
                logger.info(f'Treating unsupported language tag "{subtitle.audio_language}" as English')
            else:
                logger.info(f'Unsupported audio language tag: "{subtitle.audio_language}"')
                subtitle.content = None
                return

        logger.info(f'WhisperAI: Starting {subtitle.task} from {subtitle.audio_language} ({language_from_alpha3(subtitle.audio_language)}) -> {output_language} ({language_from_alpha3(output_language)}) - File: "{os.path.basename(subtitle.video.original_path)}"')

        start_time = time.time()

        video_name = subtitle.video.original_path if self.pass_video_name else None
        response = self.session.post(
            f"{self.endpoint}/asr",
            params={
                'task': subtitle.task,
                'language': input_language,
                'output': 'srt',
                'encode': 'false',
                'video_file': video_name
            },
            files={'audio_file': out},
            timeout=(self.response, self.timeout)
        )
        # for debugging, log if anything got returned
        subtitle_length = len(response.content)
        logger.debug(f'Returned subtitle length is {subtitle_length:,} bytes')
        subtitle_length = min(subtitle_length, 1000)
        if subtitle_length > 0:
            logger.debug(f'First {subtitle_length} bytes of subtitle: {response.content[0:subtitle_length]}')

        subtitle.content = response.content
        logger.info(f'WhisperAI: Completed {subtitle.task} from {subtitle.audio_language} ({language_from_alpha3(subtitle.audio_language)}) -> {output_language} ({language_from_alpha3(output_language)}) - Duration: {timedelta(seconds=round(time.time() - start_time))} - File: "{os.path.basename(subtitle.video.original_path)}"')
