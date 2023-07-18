import { expectType, expectAssignable  } from 'tsd';
import { Append, AppendTwo, Length, SplitAt, SliceTo, StartsWith, TakeWhile } from './';

expectType<Append<"abc", "def">>("abcdef")
expectAssignable<Append<"abc", "d" | "e">>("abcd")
expectAssignable<Append<"abc", "d" | "e">>("abce")

expectType<AppendTwo<"abc", "def", "ghi">>("abcdefghi")
expectAssignable<AppendTwo<"abc", "d" | "e", "fgh">>("abcdfgh")
expectAssignable<AppendTwo<"abc", "d" | "e", "fgh">>("abcefgh")

expectType<Length<"aaaaaaaaaaaassssss">>(18)

expectType<SplitAt<"1.223", ".">>(["1", ".", "223"])
expectType<SplitAt<"22.223", ".">>(["22", ".", "223"])

expectType<SliceTo<"1", 3>>("1")
expectType<SliceTo<"1245", 3>>("124")
expectType<SliceTo<"1245", 5>>("1245")

expectType<StartsWith<"abcd", "ab">>(["ab", "cd"])
expectType<StartsWith<"abcd", "d">>(false)
expectAssignable<StartsWith<"abcd", "ab" | "bc">>(["ab", "cd"])
expectAssignable<StartsWith<"abcd", "ab" | "a">>(["ab", "cd"])
expectAssignable<StartsWith<"abcd", "ab" | "a">>(["a", "bcd"])

expectType<TakeWhile<"aabc", "a">>("aa")
expectType<TakeWhile<"aabc", "a" | "b">>("aab")
expectType<TakeWhile<"aabc", "q">>("")