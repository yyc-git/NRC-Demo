import * as Num from "../number/index";
import * as StrNum from "../strnum/index"

/**
 * Convenience type to represent the concatenation of two string types.
 * @example
 * Append<'abc', 'def'> => 'abcdef'
 * Append<'abc', 'd' | 'e'> => 'abcd' | 'abce'
 */
export type Append<Start extends string, End extends string> = `${Start}${End}`;

/**
 * Convenience type to represent the concatenation of three string types.
 * @example
 * AppendTwo<'abc', 'def', 'ghi'> => 'abcdefghi'
 * AppendTwo<'abc', 'd' | 'e', 'fgh'> => 'abcdfgh' | 'abcefgh'
 */
export type AppendTwo<Start extends string, Middle extends string, End extends string> = `${Start}${Middle}${End}`;

/**
 * Returns the length of the given string S.
 * @example
 * Length<'abc'> => 3
 */
export type Length<S extends string, Result extends number = 0> = S extends '' ? Result : S extends `${string}${string}${string}${string}${string}${string}${string}${string}${string}${string}${infer Rest}` ? Length<Rest, Num.Add<Result, 10>> : S extends `${string}${string}${string}${string}${string}${infer Rest}` ? Length<Rest, Num.Add<Result, 5>> : S extends `${string}${infer Rest}` ? Length<Rest, Num.Add<Result, 1>> : never;

/**
 * Returns never if the given string is empty, otherwise the given string.
 */
export type NonEmptyString<S extends string> = '' extends S ? never : unknown;

/**
 * Returns false if the given string does not contain the given `Middle` type,
 * or a 3-tuple containing the start, the matched middle, and the rest.
 */
export type SplitAt<S extends string, Middle extends string & NonEmptyString<Middle>> = S extends AppendTwo<infer Start, Middle, infer End> ? S extends AppendTwo<Start, infer MiddleInstance, End> ? [Start, MiddleInstance, End] : false : ['', '', S];

export type First<S extends string> = S extends Append<infer First, string> ? First : never;

type _SliceTo<S extends string, EndIndex extends string, Result extends string> = EndIndex extends '0' ? Result : S extends Append<string, infer Rest> ? _SliceTo<Rest, StrNum.Decr<EndIndex>, Append<Result, First<S>>> : never;

export type SliceTo<S extends string, EndIndex extends number> = Num.LessThan<Length<S>, EndIndex> extends true ? S : _SliceTo<S, StrNum.FromNumber<EndIndex>, "">



/**
 * If the given string does not start with the given `Start` type, returns false.
 * Otherwise, returns a tuple containing the matched start and the rest.
 */
export type StartsWith<
  S extends string,
  Start extends string & NonEmptyString<Start>
> = S extends Append<Start, infer Rest>
  ? S extends Append<infer StartInstance, Rest>
    ? [StartInstance, Rest]
    : false
  : false;


/**
 * Returns part of the string as long as its parts match Sub.
 */
export type TakeWhile<
  S extends string,
  Sub extends string & NonEmptyString<Sub>,
  Result extends string = ''
> = StartsWith<S, Sub> extends [infer SubInstance, infer Rest]
  ? TakeWhile<string & Rest, Sub, Append<Result, string & SubInstance>>
  : Result;
