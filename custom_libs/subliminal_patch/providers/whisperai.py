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

import copy
import os
import subprocess
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
    "und": "eng",
    "enc": "eng", # Some older files like e.g. DivX are using ISO 639-1 language codes like e.g. "en" for english, which gets wrongly assigned to the language "En" with the ISO 639-3 code "enc"
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
def get_ISO_639_3_code(iso639_2_code):
    ISO_639_2_TO_3_MAPPING = {
        'fre': 'fra',  # French
        'ger': 'deu',  # German
        'dut': 'nld',  # Dutch
        'gre': 'ell',  # Greek
        'chi': 'zho',  # Chinese
        'cze': 'ces',  # Czech
        'ice': 'isl',  # Icelandic
        'rum': 'ron',  # Romanian
        'slo': 'slk',  # Slovak
        'alb': 'sqi',  # Albanian
        'arm': 'hye',  # Armenian
        'baq': 'eus',  # Basque
        'bur': 'mya',  # Burmese
        'geo': 'kat',  # Georgian
        'mac': 'mkd',  # Macedonian
        'may': 'msa',  # Malay
        'per': 'fas',  # Persian
        'wel': 'cym',  # Welsh
    }

    # Convert ISO 639-2 bibliographic code to ISO 639-3 if it's in our mapping
    if iso639_2_code in ISO_639_2_TO_3_MAPPING:
        iso639_3_code = ISO_639_2_TO_3_MAPPING[iso639_2_code]
        logger.debug(f'Converting language code from "{iso639_2_code}" to "{iso639_3_code}"')
        return iso639_3_code

    # If it's not in our mapping, try to look it up directly
    try:
        language = languages.get(bibliographic=iso639_2_code)
        if language:
            logger.debug(f'Found language with bibliographic code "{iso639_2_code}": "{language.alpha_3}"')
            return language.alpha_3
    except:
        pass

    # If all else fails, return the original code
    return iso639_2_code


@functools.lru_cache(2)
def encode_audio_stream(path, ffmpeg_path, audio_stream_language=None, stream_index=None):
    if audio_stream_language:
        audio_stream_language = get_ISO_639_2_code(audio_stream_language)

    if stream_index is None:
        logger.debug(f'Encoding first audio stream (a:0) to WAV with ffmpeg in "{os.path.basename(path)}"')
    else:
        logger.debug(f'Encoding audio stream #{stream_index} to WAV with ffmpeg in "{os.path.basename(path)}"')
    try:
        lang_info = ""
        if audio_stream_language:
            try:
                lang_name = language_from_alpha3(audio_stream_language)
                lang_info = f" - Language: '{audio_stream_language}' ({lang_name})"
            except:
                lang_info = f" - Language: '{audio_stream_language}'"

        # Use ffmpeg absolute stream index syntax instead of audio-relative index
        if stream_index is not None:
            cmd = [
                ffmpeg_path, "-nostdin", "-threads", "0",
                "-i", path,
                "-map", f"0:{stream_index}",
                "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le",
                "-af", "aresample=async=1", "-ar", "16000",
                "-", "-vn", "-dn", "-sn"
            ]

            logger.debug(f'Executing FFmpeg command: {' '.join(cmd)}')

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            out, err = process.communicate()

            if process.returncode != 0:
                logger.warning(f'ffmpeg failed to load audio: {err.decode()}')
                return None

        else:
            # Use ffmpeg for language or default stream selection
            input_node = ffmpeg.input(path, threads=0)

            if audio_stream_language:
                # Handle language-based selection if needed
                iso_code = get_ISO_639_2_code(audio_stream_language)
                audio = input_node['a:m:language:' + iso_code]
                logger.debug(f'Selecting audio stream by language: "{iso_code}"')
            else:
                # Default to first audio stream
                audio = input_node['a:0']
            out, _ = (
                audio.output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000, af="aresample=async=1")
                .global_args("-vn", "-dn", "-sn")
                .run(
                    cmd=[ffmpeg_path, "-nostdin"],
                    capture_stdout=True,
                    capture_stderr=True,
                )
            )

        if stream_index is None:
            logger.debug(f'Finished encoding first audio stream (a:0) in "{os.path.basename(path)}" - Encoded audio size: {len(out):,} bytes')
        else:
            logger.debug(f'Finished encoding stream #{stream_index} in "{os.path.basename(path)}" - Encoded audio size: {len(out):,} bytes')
        return out

    except ffmpeg.Error as e:
        logger.warning(f'ffmpeg failed to load audio: {e.stderr.decode()}')
        return None
    except ValueError as e:
        logger.error(f'Stream selection error: {str(e)}')
        return None
    except subprocess.SubprocessError as e:
        logger.error(f'Subprocess error: {str(e)}')
        return None

def whisper_get_language(code, name):
    # Handle 'und' language code explicitly
    if code == "und":
        logger.warning("Undefined language code detected")
        return None
    # Whisper uses an inconsistent mix of alpha2 and alpha3 language codes
    try:
        return Language.fromalpha2(code)
    except LanguageReverseError:
        try:
            return Language.fromname(name)
        except LanguageReverseError:
            logger.error(f'Could not convert Whisper language: {code} ({name})')
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
        self.original_stream_idx = None
    @property
    def id(self):
        # Construct unique id otherwise provider pool will think
        # subtitles are all the same and drop all except the first one
        # This is important for language profiles with more than one language
        audio_lang = str(self.audio_language) if self.audio_language else "none"
        return f"{self.video.original_name}_{self.task}_{str(self.language)}_{audio_lang}"

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
            raise ConfigurationError('Whisper Web Service Connection/response timeout must be provided')

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
        logger.debug(f'Using ambiguous language codes: {self.ambiguous_language_codes}')

    def initialize(self):
        self.session = Session()
        self.session.headers['User-Agent'] = 'Subliminal/%s' % __short_version__

    def terminate(self):
        self.session.close()

    @functools.lru_cache(2048)
    def detect_language(self, path, stream_index=None) -> Language:
        if stream_index is None:
            logger.debug(f'Detecting language for first audio stream (a:0) in "{os.path.basename(path)}"')
        else:
            logger.debug(f'Detecting language for stream #{stream_index} in "{os.path.basename(path)}"')
        out = encode_audio_stream(path, self.ffmpeg_path, stream_index=stream_index)
        if out is None:
            if stream_index is None:
                logger.info(f'WhisperAI: cannot detect language of "{os.path.basename(path)}" (first audio stream (a:0)) -> missing/bad audio stream!')
            else:
                logger.info(f'WhisperAI: cannot detect language of "{os.path.basename(path)}" (stream #{stream_index}) -> missing/bad audio stream!')
            return None

        try:
            video_name = path if self.pass_video_name else None
            if stream_index is None:
                logger.debug(f'Sending {len(out):,} bytes of audio data to Whisper for language detection (first audio stream a:0) in "{os.path.basename(path)}"')
            else:
                logger.debug(f'Sending {len(out):,} bytes of audio data to Whisper for language detection (stream #{stream_index}) in "{os.path.basename(path)}"')
            r = self.session.post(f"{self.endpoint}/detect-language",
                                params={'encode': 'false', 'video_file': {video_name}},
                                files={'audio_file': out},
                                timeout=(self.response, self.timeout))
            results = r.json()

        except (JSONDecodeError, requests.exceptions.JSONDecodeError) as e:
            logger.error(f'Invalid JSON response in language detection: {e}')
            return None
        except KeyError as e:
            logger.error(f'KeyError in language detection response: {e}')
            return None
        except Exception as e:
            logger.error(f'Unexpected error in language detection: {e}')
            return None

        if not results.get("language_code"):
            logger.info(f'Whisper returned empty language code for stream #{stream_index}')
            return None

        # Explicitly handle 'und' from Whisper results
        if results["language_code"] == "und":
            if stream_index is None:
                logger.info(f'Whisper detected undefined language for first audio stream (a:0) in "{os.path.basename(path)}"')
                logger.debug(f'Whisper detection raw results for first audio stream (a:0) in "{os.path.basename(path)}": {results}')
            else:
                logger.info(f'Whisper detected undefined language for stream #{stream_index} in "{os.path.basename(path)}"')
                logger.debug(f'Whisper detection raw results for stream #{stream_index} in "{os.path.basename(path)}": {results}')
            return None

        if stream_index is None:
            logger.debug(f'Whisper detection raw results for first audio stream (a:0) in "{os.path.basename(path)}": {results}')
        else:
            logger.debug(f'Whisper detection raw results for stream #{stream_index} in "{os.path.basename(path)}": {results}')

        return whisper_get_language(results["language_code"], results["detected_language"])

    def query(self, language, video, original_stream_idx=None):
        logger.debug(f'Processing subtitle request: {language.alpha3} ({language_from_alpha3(language.alpha3)}) for "{os.path.basename(video.original_path)}"')

        if language not in self.languages:
            logger.debug(f'Language {language.alpha3} not supported by Whisper')
            return None

        sub = WhisperAISubtitle(language, video)
        sub.task = "transcribe"
        if original_stream_idx is not None:
            sub.original_stream_idx = original_stream_idx
        # Handle undefined/no audio languages
        if not video.audio_languages:
            logger.debug(f'No audio language tags present in "{os.path.basename(video.original_path)}" -> forcing detection!')
            detected_lang = self.detect_language(video.original_path, stream_index=original_stream_idx)
            if not detected_lang:
                sub.task = "error"
                sub.release_info = "Language detection failed"
                return sub

            # Apply language mapping after detection
            detected_alpha3 = detected_lang.alpha3
            if detected_alpha3 in language_mapping:
                detected_alpha3 = language_mapping[detected_alpha3]
                logger.debug(f'Mapped detected language "{detected_lang.alpha3}" -> "{detected_alpha3}" in "{os.path.basename(video.original_path)}')

            sub.audio_language = detected_alpha3

            # Determine if we need transcription or translation
            if detected_alpha3 != language.alpha3:
                # Set to translation task only if target is English
                if language.alpha3 == "eng":
                    sub.task = "translate"
                else:
                    # Non-English target languages aren't supported for translation
                    if sub.original_stream_idx is None:
                        logger.debug(
                            f'Cannot translate from first audio stream (a:0) ({detected_alpha3} -> {language.alpha3})! '
                            f'Only translations to English supported! File: "{os.path.basename(video.original_path)}"'
                        )
                    else:
                        logger.debug(
                            f'Cannot translate from audio stream #{sub.original_stream_idx} ({detected_alpha3} -> {language.alpha3})! '
                            f'Only translations to English supported! File: "{os.path.basename(video.original_path)}"'
                        )
                    return None
        else:
            # Process all audio languages with mapping
            processed_languages = []
            for lang in video.audio_languages:
                processed_lang = language_mapping.get(lang, lang)
                processed_languages.append(processed_lang)
                if lang != processed_lang:
                    logger.debug(f'Mapping audio language tag: {lang} -> {processed_lang} in "{os.path.basename(video.original_path)}"')

            # Check for direct match between requested language and processed audio languages
            matched = False
            for idx, processed_lang in enumerate(processed_languages):
                if language.alpha3 == processed_lang:
                    sub.audio_language = processed_lang
                    original_lang = video.audio_languages[idx]
                    if len(video.audio_languages) > 1:
                        sub.force_audio_stream = original_lang
                    matched = True
                    break

            if not matched:
                sub.task = "translate"
                sub.audio_language = processed_languages[0] if processed_languages else None

            if not sub.audio_language:
                sub.task = "error"
                sub.release_info = "No valid audio language determined"
                return sub
            else:
                # Handle case where audio language exists but language code is on "ambiguous_language_codes"
                original_ambiguous = any(
                    lang in self.ambiguous_language_codes
                    for lang in video.audio_languages
                )

                # In the query method, when handling ambiguous language codes, store the original language first
                if original_ambiguous:
                    # Store original language code BEFORE any processing
                    original_lang_code = video.audio_languages[0] if video.audio_languages else "und"
                    try:
                        original_lang_name = language_from_alpha3(original_lang_code)
                    except:
                        original_lang_name = "Undefined"

                    # Format audio languages with both code and name
                    formatted_audio_langs = [
                        f'{lang} ({language_from_alpha3(lang)})'
                        for lang in video.audio_languages
                    ]

                    file_idx = sub.original_stream_idx if sub.original_stream_idx is not None else "unknown"
                    logger.debug(
                        f'Audio language tag {", ".join(formatted_audio_langs)} from audio stream #{file_idx} '
                        f'matches "Ambiguous Languages Codes" list: {self.ambiguous_language_codes} -> forcing detection!'
                    )

                    detected_lang = self.detect_language(video.original_path, stream_index=sub.original_stream_idx)
                    if detected_lang is None:
                        sub.task = "error"
                        sub.release_info = "bad/missing audio stream - cannot transcribe"
                        return sub

                    detected_alpha3 = detected_lang.alpha3

                    # Apply language mapping after detection
                    if detected_alpha3 in language_mapping:
                        detected_alpha3 = language_mapping[detected_alpha3]

                    sub.audio_language = detected_alpha3
                    sub.task = "transcribe" if detected_alpha3 == language.alpha3 else "translate"

                    # Simplify the log message and make it match the desired format
                    file_idx = sub.original_stream_idx if sub.original_stream_idx is not None else "unknown"
                    logger.debug(
                        f'WhisperAI detected language for audio stream #{file_idx}: {original_lang_code} ({original_lang_name}) -> '
                        f'{detected_lang.alpha3} ({language_from_alpha3(detected_lang.alpha3)}) - '
                        f'Requested Subtitles: {language.alpha3} ({language_from_alpha3(language.alpha3)})'
                    )
                else:
                    formatted_original = [
                        f'{lang} ({language_from_alpha3(lang)})'
                        for lang in video.audio_languages
                    ]

                    file_idx = sub.original_stream_idx if sub.original_stream_idx is not None else "unknown"

                    # Only log original language tag if it differs from the first logged language tag
                    first_logged_tag = f'{sub.audio_language} ({language_from_alpha3(sub.audio_language)})'
                    original_tags = ", ".join(formatted_original)

                    if first_logged_tag != original_tags:
                        logger.debug(
                            f'Using language tag for audio stream #{file_idx}: {first_logged_tag} in "{os.path.basename(video.original_path)}" - '
                            f'Original language tag: {original_tags}'
                        )
                    else:
                        logger.debug(
                            f'Using existing language tag from audio stream #{file_idx}: {first_logged_tag} in "{os.path.basename(video.original_path)}"'
                        )

            if sub.task == "translate":
                if language.alpha3 != "eng":
                    file_idx = sub.original_stream_idx if sub.original_stream_idx is not None else "unknown"

                    logger.debug(
                        f'Cannot translate from audio stream #{file_idx} ({sub.audio_language} -> {language.alpha3})! '
                        f'Only translations to English supported! File: "{os.path.basename(sub.video.original_path)}"'
                    )
                    return None

        sub.release_info = f'{sub.task} {language_from_alpha3(sub.audio_language)} audio -> {language_from_alpha3(language.alpha3)} SRT'
        logger.debug(f'WhisperAI query task: {sub.task}: {sub.audio_language} ({language_from_alpha3(sub.audio_language)}) -> {language.alpha3} ({language_from_alpha3(language.alpha3)}) - File: ({video.original_path})')
        return sub

    def list_subtitles(self, video, languages):
        logger.debug(f'Languages requested from WhisperAI: {", ".join(f"{l.alpha3} ({language_from_alpha3(l.alpha3)})" for l in languages)} - File: "{os.path.basename(video.original_path)}"')

        if not video.audio_languages:
            # If no audio languages, use the existing logic
            subtitles = [self.query(l, video) for l in languages]
            return [s for s in subtitles if s is not None]

        # Attempt to get correct stream indices by probing the file
        actual_indices = {}
        file_audio_streams = []

        try:
            # Fix ffprobe path construction
            if '/' in self.ffmpeg_path:
                ffprobe_path = os.path.dirname(self.ffmpeg_path) + '/ffprobe'
            else:
                ffprobe_path = 'ffprobe'

            logger.debug(f'Using ffprobe path: "{ffprobe_path}"')
            probe = ffmpeg.probe(video.original_path, cmd=ffprobe_path)

            # Extract all audio streams with their actual file indices
            for stream in probe['streams']:
                if stream.get('codec_type') == 'audio':
                    stream_idx = int(stream['index'])
                    stream_lang = stream.get('tags', {}).get('language', 'und')

                    # Convert ISO 639-2 bibliographic to ISO 639-3 if needed
                    stream_lang = get_ISO_639_3_code(stream_lang)

                    file_audio_streams.append((stream_idx, stream_lang))

            # Log all streams with their file indices
            if file_audio_streams:
                stream_info = []
                for file_idx, lang in file_audio_streams:
                    try:
                        lang_name = language_from_alpha3(lang)
                    except:
                        lang_name = "Unknown"
                    stream_info.append(f'Audio stream {file_idx}: {lang} ({lang_name})')
                logger.debug(f'All audio streams in "{os.path.basename(video.original_path)}":\n' + '\n'.join(stream_info))

            logger.debug(f'Full file audio stream list: {file_audio_streams}')

            # Map each language in Bazarr's filtered list to its actual file index
            for bazarr_idx, lang in enumerate(video.audio_languages):
                for file_idx, file_lang in file_audio_streams:
                    if file_lang == lang:
                        actual_indices[bazarr_idx] = file_idx
                        logger.debug(f'Found first matching language tag: {lang} ({language_from_alpha3(lang)}) at audio stream #{file_idx} in "{os.path.basename(video.original_path)}"')
                        break
                else:
                    # Fallback if language not found - use first available audio stream from file
                    if file_audio_streams:
                        first_audio_stream_idx = file_audio_streams[0][0]
                        actual_indices[bazarr_idx] = first_audio_stream_idx
                        logger.warning(f'Could not find language tag "{lang}" in audio streams, using first available audio stream #{first_audio_stream_idx} in "{os.path.basename(video.original_path)}"')
                    else:
                        logger.error(f'No audio streams found in file, cannot process')
                        actual_indices[bazarr_idx] = -1  # Invalid index to indicate failure
        except Exception as e:
            logger.warning(f'Unable to probe file for accurate stream indices: {e}')
            # Default to using the bazarr-provided indices
            actual_indices = {idx: idx for idx, _ in enumerate(video.audio_languages)}

            # If we couldn't probe the file, still log the audio streams with Bazarr indices
            if video.audio_languages:
                stream_info = []
                for idx, lang in enumerate(video.audio_languages):
                    try:
                        lang_name = language_from_alpha3(lang)
                    except:
                        lang_name = "Unknown"
                    stream_info.append(f'Audio stream #{idx} (Bazarr index): {lang} ({lang_name})')
                logger.debug(f'All audio streams in media file (using Bazarr indices):\n' + '\n'.join(stream_info))

        all_subtitles = []

        # Process unique languages using the corrected stream indices
        seen_langs = set()
        for bazarr_idx, lang in enumerate(video.audio_languages):
            if lang not in seen_langs:
                seen_langs.add(lang)
                # Use the actual file index
                file_idx = actual_indices.get(bazarr_idx, bazarr_idx)

                try:
                    lang_name = language_from_alpha3(lang)
                except:
                    lang_name = "Unknown"
                logger.debug(f'Processing audio stream #{file_idx} with language tag: {lang} ({lang_name}) in "{os.path.basename(video.original_path)}"')

                # Create a working copy of the video with just this language
                video_copy = copy.copy(video)
                video_copy.audio_languages = [lang]

                # Query each requested subtitle language for this audio stream
                for l in languages:
                    subtitle = self.query(l, video_copy, original_stream_idx=file_idx)
                    if subtitle is not None:
                        all_subtitles.append(subtitle)

        return all_subtitles

    def download_subtitle(self, subtitle: WhisperAISubtitle):
        # Invoke Whisper through the API. This may take a long time depending on the file.
        # TODO: This loads the entire file into memory, find a good way to stream the file in chunks

        # TODO: Remove this part since at line ~473 this case is already handeled?
        if subtitle.task == "translate" and subtitle.language.alpha3 != "eng":
            logger.warning(f'WhisperAI cannot translate to non-English target language: {subtitle.language.alpha3}')
            subtitle.content = None
            return
        # TODO: Remove this part since at line ~508 this case is already handeled?
        if subtitle.task == "error":
            return

        out = encode_audio_stream(
            subtitle.video.original_path,
            self.ffmpeg_path,
            audio_stream_language=subtitle.force_audio_stream,
            stream_index=subtitle.original_stream_idx
        )
        # TODO: Remove this part since at line ~391 this case is already handeled?
        if not out:
            logger.info(f'WhisperAI cannot process "{subtitle.video.original_path}" due to missing/bad audio stream')
            subtitle.content = None
            return

        logger.debug(f'Audio stream length: {len(out):,} bytes')

        output_language = "eng" if subtitle.task == "translate" else subtitle.audio_language

        input_language = whisper_get_language_reverse(subtitle.audio_language)
        # TODO: change this part? Unknown language tags can be mapped with "language_mapping" (line ~146). Should all other unknown (and not mapped) languages be treated as English by default?
        if not input_language:
            if output_language == "eng":
                input_language = "en"
                subtitle.task = "transcribe"
                logger.info(f'WhisperAI: Treating unsupported audio language tag {subtitle.audio_language} ({language_from_alpha3(subtitle.audio_language)}) as English in "{os.path.basename(subtitle.video.original_path)}"')
            else:
            # TODO: Remove this part since at line ~441 this case is already handeled?
                logger.info(f'WhisperAI: Unsupported audio language tag {subtitle.audio_language} ({language_from_alpha3(subtitle.audio_language)}) in "{os.path.basename(subtitle.video.original_path)}"')
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
