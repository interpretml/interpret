# Copyright (c) 2023 The InterpretML Contributors
# Distributed under the MIT software license

from math import floor


def normalize_initial_seed(seed):
    # We want cross-language seeds, so we cannot use 64-bit integers since those are not supported
    # on all platforms. Unsigned integers are also not supported on many platforms. Most platforms will
    # either support 32-bit signed integers, or float64 values. Float64 values can represent all 32-bit integers
    # exactly, so limiting our seeds to 32-bit integers allows us to support the widest number of platforms.
    #
    # If you want cross-platform results, use seeds in the range: -2147483648 <= seed and seed <= 21474836487
    #
    # But, if the platform supports 64-bit integers or for languages like python where integers have no limits,
    # then we'd rather not throw an exception if something else is provided. We can convert non-portable seeds
    # into something that fits our API by taking modulos to force them into our cross-platform range
    #
    # We use a simple conversion because we use the same method in multiple languages,
    # and we need to keep the results identical between them, so simplicity is key.
    #
    # The result of the modulo operator is not standardized accross languages for negative numbers.
    # We can use rounding on divisions to get standardized modulos, and this will work with float64 inputs too.
    # https://torstencurdt.com/tech/posts/modulo-of-negative-numbers

    if seed is None:
        return None  # non-deterministic random numbers
    if 2147483647 < seed:
        # we'd like to modulo by 2147483648 but that is not a legal 32-bit signed int
        # if the user provides 2147483647 exactly then we use it, but otherwise our range is restricted
        # add one to prevent generating 0 because we generate 0 for the negative multiples of -2147483648
        return int(round(seed - floor(seed / 2147483647) * 2147483647)) + 1
    if seed < -2147483648:
        # we'd like to modulo by -2147483649 but that is not a legal 32-bit signed int
        # if the user provides -2147483648 exactly then we use it, but otherwise our range is restricted
        return int(round(seed - floor(seed / -2147483648) * -2147483648))
    return seed


def increment_seed(seed):
    # in places where we need to pass seeds, and we want user repetability, like the EBMPreprocessor
    # we can increment the seed instead of generating a random and human-unpredictable seed
    if seed is None:
        return None
    if seed == 2147483647:
        return -2147483648
    return seed + 1
