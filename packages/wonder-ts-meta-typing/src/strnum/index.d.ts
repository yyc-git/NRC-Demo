import * as Str from "../str/index";
import * as U from "../utils/index";

/**
 * Positive numeric digits as strings
 */
export type PosDigit = '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9';

/**
 * All numeric digits as strings
 */
export type Digit = '0' | PosDigit;

/**
 * Type that will return the incoming type if the value is a valid positive integer,
 * or `never` otherwise.
 */
// export type PosNum<N extends string> =
// 	N extends `-${number}` ? never : N;
export type PosNum<N extends string> = N extends Str.Append<PosDigit, infer Rest> ? Rest extends Str.TakeWhile<Rest, Digit> ? N : never : never;

/**
 * Type that will return the incoming type if the value is a valid natural number,
 * or `never` otherwise.
 * @example
 * NatNum<'321'> => '321'
 * NatNum<'0'> => '0'
 */
// export type NatNum<N extends string> = N extends '0' ? '0' : PosNum<N>;
export type NatNum<N extends string> = N extends '0' ? '0' : PosNum<N>;

/**
 * Converts a natural number to a string-number, otherwise never.
 * @example
 * FromNumber<123> => '123'
 * FromNumber<-13> => never
 */
export type FromNumber<N extends number> = NatNum<`${N}`>;

/**
 * A symmetric tuple, where the order or the elements does not matter.
 * As an example: SymTup<'a', 1> is equal to SymTup<1, 'a'>
 */
export type SymTup<A, B> = [A, B] | [B, A];
/**
 * Given two string digits, this type returns a tuple of which the first element
 * is the digit resulting from addition of the two digits, and the second element
 * is a boolean indicating whether there was an overflow.
 * @example
 * AddDigit<'1', '2'> => ['3', false]
 * AddDigit<'8', '4'> => ['2', true]
 */
export type AddDigit<D1 extends Digit, D2 extends Digit> = D1 extends '0' ? [D2, false] : D2 extends '0' ? [D1, false] : [D1, D2] extends SymTup<'1', '9'> | SymTup<'2', '8'> | SymTup<'3', '7'> | SymTup<'4', '6'> | '5'[] ? ['0', true] : [D1, D2] extends SymTup<'2', '9'> | SymTup<'3', '8'> | SymTup<'4', '7'> | SymTup<'5', '6'> ? ['1', true] : [D1, D2] extends '1'[] ? ['2', false] : [D1, D2] extends SymTup<'3', '9'> | SymTup<'4', '8'> | SymTup<'5', '7'> | '6'[] ? ['2', true] : [D1, D2] extends SymTup<'1', '2'> ? ['3', false] : [D1, D2] extends SymTup<'4', '9'> | SymTup<'5', '8'> | SymTup<'6', '7'> ? ['3', true] : [D1, D2] extends SymTup<'1', '3'> | '2'[] ? ['4', false] : [D1, D2] extends SymTup<'5', '9'> | SymTup<'6', '8'> | '7'[] ? ['4', true] : [D1, D2] extends SymTup<'1', '4'> | SymTup<'2', '3'> ? ['5', false] : [D1, D2] extends SymTup<'6', '9'> | SymTup<'7', '8'> ? ['5', true] : [D1, D2] extends SymTup<'1', '5'> | SymTup<'2', '4'> | '3'[] ? ['6', false] : [D1, D2] extends SymTup<'7', '9'> | '8'[] ? ['6', true] : [D1, D2] extends SymTup<'1', '6'> | SymTup<'2', '5'> | SymTup<'3', '4'> ? ['7', false] : [D1, D2] extends SymTup<'8', '9'> ? ['7', true] : [D1, D2] extends SymTup<'1', '7'> | SymTup<'2', '6'> | SymTup<'3', '5'> | '4'[] ? ['8', false] : [D1, D2] extends '9'[] ? ['8', true] : [D1, D2] extends SymTup<'1', '8'> | SymTup<'2', '7'> | SymTup<'3', '6'> | SymTup<'4', '5'> ? ['9', false] : never;

/**
 * Returns the result of adding the two given string-numbers.
 * @example
 * Add<'13', '8'> => '21'
 * Add<'139', '5232'> => '5371'
 */
export type Add<N1 extends string, N2 extends string> = N1 extends Str.Append<infer N1Start, Digit> ? N2 extends Str.Append<infer N2Start, Digit> ? N1 extends Str.Append<N1Start, infer N1LastDigit> ? N2 extends Str.Append<N2Start, infer N2LastDigit> ? AddDigit<N1LastDigit & Digit, N2LastDigit & Digit> extends [
	infer NewDigit,
	infer Overflow
] ? Overflow extends true ? [
	N1Start,
	N2Start
] extends ['', ''] ? Str.Append<'1', string & NewDigit> : Str.Append<Add<Add<N1Start, N2Start>, '1'>, NewDigit & string> : Str.Append<Add<N1Start, N2Start>, NewDigit & string> : never : never : never : N1 : N2;

export type IsWhole<N extends string> =
	N extends `${number}.${number}` ? false : true;

type _AddWholeAndFloat<NWhole extends string, NFloat extends string> = // NFloat extends Str.AppendTwo<infer NFloatWhole, '.', infer _> ?
	// 	NFloat extends Str.AppendTwo<NFloatWhole, '.', infer NFloatDecimal> ?
	// 	Str.AppendTwo<Add<NWhole, NFloatWhole>, '.', NFloatDecimal> : never : never;
	Str.SplitAt<NFloat, '.'> extends [infer NFloatWhole, '.', infer NFloatDecimal] ? Str.AppendTwo<Add<NWhole, NFloatWhole & string>, '.', NFloatDecimal & string> : never

/*!TODO rename to _Decr will cause compile fail, why?*/
type Decr<N extends string> = Subtract<N, '1'>;

type _Equal<N1 extends string, N2 extends string> = N1 extends N2 ? N2 extends N1 ? true : false : false;

type _LessThan<N1 extends string, N2 extends string> = Subtract<N1, N2> extends never ? true : false;

type _Max<N1 extends string, N2 extends string> = N1 extends never ? never : N2 extends never ? never : Subtract<N1, N2> extends never ? N2 : N1;

export type MinFloat<N1 extends string, N2 extends string> = N1 extends never ? never : N2 extends never ? never : SubtractFloat<N1 & U.Check<N1>, N2 & U.Check<N2>> extends never ? N1 : N2;

export type MaxFloat<N1 extends string, N2 extends string> = N1 extends never ? never : N2 extends never ? never : SubtractFloat<N1, N2> extends never ? N2 : N1;

export type LessThanFloat<N1 extends string, N2 extends string> = SubtractFloat<N1, N2> extends never ? true : false;

export type LessThanOrEqualFloat<N1 extends string, N2 extends string> = N1 extends N2 ? true : LessThanFloat<N1, N2>;

export type GreaterThanOrEqualFloat<N1 extends string, N2 extends string> = SubtractFloat<N1, N2> extends never ? false : true;

export type GreaterThanFloat<N1 extends string, N2 extends string> = N1 extends N2 ? false : GreaterThanOrEqualFloat<N1, N2>;

export type InRangeFloat<N extends string, L extends string, H extends string> = GreaterThanOrEqualFloat<N, L> & LessThanOrEqualFloat<N, H> extends never ? false : true;

type _AlignWithZero<N extends string, ZeroCount extends string = FromNumber<Str.Length<N>>, Result extends string = ""> = {
	finish: Str.Append<N, Result>;
	next: _AlignWithZero<N, Decr<ZeroCount>, Str.Append<Result, '0'>>
}[ZeroCount extends '0' ? 'finish' : "next"];

type _AlignDecimal<N1 extends string, N2 extends string> = {
	isAlignN1: [_AlignWithZero<N1, Subtract<FromNumber<Str.Length<N2>>, FromNumber<Str.Length<N1>>>>, N2];
	isAlignN2: [N1, _AlignWithZero<N2, Subtract<FromNumber<Str.Length<N1>>, FromNumber<Str.Length<N2>>>>];
	notAlign: [N1, N2];
}[_Equal<FromNumber<Str.Length<N1>>, FromNumber<Str.Length<N2>>> extends true ? 'notAlign' :
	_LessThan<FromNumber<Str.Length<N1>>, FromNumber<Str.Length<N2>>> extends true ? 'isAlignN1' : 'isAlignN2']


// type _GetFirstDigit<N extends string> = N extends Str.Append<Digit, infer NLast> ? N extends Str.Append<infer NStart, NLast> ? NStart & Digit : never : never;

// export type _GetLastDigit<N extends string> = N extends Str.Append<infer NStart, Digit> ? N extends Str.Append<NStart, infer NLast> ? NLast & Digit : never : never;

type _isOverflow<AddResult extends string, MaxLength extends string> = _LessThan<MaxLength, FromNumber<Str.Length<AddResult>>>;

type _AddNat<N1 extends string, N2 extends string, AddResult extends string> =
	_isOverflow<AddResult, _Max<FromNumber<Str.Length<N1>>, FromNumber<Str.Length<N2>>>> extends true ? AddResult extends Str.Append<'1', infer Rest> ? [Rest & string, true] : never : [AddResult, false];

type _AddDecimal<N1 extends string, N2 extends string> =
	_AlignDecimal<N1, N2> extends [infer AlignedN1, infer AlignedN2] ?
	_AddNat<AlignedN1 & string, AlignedN2 & string, Add<AlignedN1 & string, AlignedN2 & string>> : never

type _ExtractWhole<N extends string> = IsWhole<N> extends true ? N : Str.SplitAt<N, '.'> extends [infer Whole, '.', infer _] ? Whole & string : never;

type _ExtractDecimal<N extends string> = IsWhole<N> extends true ? '0' : Str.SplitAt<N, '.'> extends [infer _, '.', infer Decimal] ? Decimal & string : never;

type _FloorDecimal<N extends string, Precision extends number> = Str.SliceTo<N, Precision>

type _AddFloatAndFloat<N1 extends string, N2 extends string, AddWholeResult extends string, Precision extends number> = _AddDecimal<_FloorDecimal<_ExtractDecimal<N1>, Precision>, _FloorDecimal<_ExtractDecimal<N2>, Precision>> extends [infer Result, infer Overflow] ? Overflow extends true ? Str.AppendTwo<Add<AddWholeResult, '1'>, '.', Result & string> : Str.AppendTwo<AddWholeResult, '.', Result & string> : never

// type _RemoveZeroDecimal<N extends string> = N extends Str.AppendTwo<infer Whole, '.', '0'> ? Whole : N
// type _RemoveZeroDecimal<N extends string> = N;

export type AddFloat<N1 extends string, N2 extends string, Precision extends number = 3> = {
	isN1Whole: _AddWholeAndFloat<N1, N2>;
	isN2Whole: _AddWholeAndFloat<N2, N1>;
	isAllWhole: Add<N1, N2>;
	isAllNotWhole: _AddFloatAndFloat<N1, N2, Add<_ExtractWhole<N1>, _ExtractWhole<N2>>, Precision>;
}[IsWhole<N1> | IsWhole<N2> extends true ? 'isAllWhole' : IsWhole<N1> extends true ? 'isN1Whole' : IsWhole<N2> extends true ? 'isN2Whole' : 'isAllNotWhole']

/**
 * Given two string digits, returns a tuple of which the first element is the resulting digit from subtracting the second from the first,
 * and the second element a boolean that is true if there is an 'underflow' or borrow, never otherwise.
 * @example
 * SubDigit<'5', '3'> => ['2', never]
 * SubDigit<'3', '6'> => ['7', true]
 */
export type SubDigit<D1 extends Digit, D2 extends Digit> = D2 extends '0' ? [D1, false] : D1 extends D2 ? ['0', false] : Str.Append<D1, D2> extends infer DD ? DD extends '09' | '21' | '32' | '43' | '54' | '65' | '76' | '87' | '98' ? ['1', U.Extends<DD, '09'>] : DD extends '08' | '19' | '31' | '42' | '53' | '64' | '75' | '86' | '97' ? ['2', U.Extends<DD, '08' | '19'>] : DD extends '07' | '18' | '29' | '41' | '52' | '63' | '74' | '85' | '96' ? ['3', U.Extends<DD, '07' | '18' | '29'>] : DD extends '06' | '17' | '28' | '39' | '51' | '62' | '73' | '84' | '95' ? ['4', U.Extends<DD, '06' | '17' | '28' | '39'>] : DD extends '05' | '16' | '27' | '38' | '49' | '61' | '72' | '83' | '94' ? ['5', U.Extends<DD, '05' | '16' | '27' | '38' | '49'>] : DD extends '04' | '15' | '26' | '37' | '48' | '59' | '71' | '82' | '93' ? ['6', U.Extends<DD, '04' | '15' | '26' | '37' | '48' | '59'>] : DD extends '03' | '14' | '25' | '36' | '47' | '58' | '69' | '81' | '92' ? ['7', U.Extends<DD, '03' | '14' | '25' | '36' | '47' | '58' | '69'>] : DD extends '02' | '13' | '24' | '35' | '46' | '57' | '68' | '79' | '91' ? [
	'8',
	U.Extends<DD, '02' | '13' | '24' | '35' | '46' | '57' | '68' | '79'>
] : DD extends '01' | '12' | '23' | '34' | '45' | '56' | '67' | '78' | '89' ? ['9', true] : never : never;

/**
 * Returns the result of subtracting the second from the first given string-number,
 * or never if the second value is greater than the first. (Only natural numbers currently supported)
 */
export type Subtract<N1 extends string, N2 extends string> = N1 extends N2 ? '0' : N1 extends Str.Append<infer N1Start, Digit> ? N2 extends Str.Append<infer N2Start, Digit> ? N1 extends Str.Append<N1Start, infer N1LastDigit> ? N2 extends Str.Append<N2Start, infer N2LastDigit> ? SubDigit<Digit & N1LastDigit, Digit & N2LastDigit> extends [
	infer NewDigit,
	infer Underflow
] ? Underflow extends true ? N1Start extends '' ? never : Subtract<Subtract<N1Start, N2Start>, '1'> extends infer Start ? Start extends '0' ? Digit & NewDigit : Str.Append<string & Start, Digit & NewDigit> : never : Subtract<N1Start, N2Start> extends infer Start ? Start extends '0' ? Digit & NewDigit : Str.Append<string & Start, Digit & NewDigit> : never : never : never : never : N1 : never;

type _SubtractWholeAndFloat<NWhole extends string, NFloat extends string> = Str.SplitAt<NFloat, '.'> extends [infer NFloatWhole, '.', infer NFloatDecimal] ? Str.AppendTwo<Subtract<NWhole, NFloatWhole & string>, '.', NFloatDecimal & string> : never

type _Align<N1 extends string, N2 extends string> = _AlignDecimal<_ExtractDecimal<N1>, _ExtractDecimal<N2>> extends [infer AlignedN1Decimal, infer AlignedN2Decimal] ? [_ExtractWhole<N1>, AlignedN1Decimal, _ExtractWhole<N2>, AlignedN2Decimal] : never

type _SubtractDecimal<N1 extends string, N2 extends string> =
	_LessThan<N1, N2> extends true ? [
		Subtract<Str.Append<'1', N1>, N2>,
		true
	] : [
		Subtract<N1, N2>,
		false
	]

type _UpdateWholeResultWithSubtractDecimalResult<WholeResult extends string, Underflow extends boolean> =
	Underflow extends false ? WholeResult : Decr<WholeResult>

export type SubtractFloat<N1 extends string, N2 extends string, Precision extends number = 3> =
	_Align<N1, N2> extends [infer N1Whole, infer AlignedN1Decimal, infer N2Whole, infer AlignedN2Decimal] ?
	_LessThan<N1Whole & string, N2Whole & string> extends true ? never :
	_SubtractDecimal<_FloorDecimal<AlignedN1Decimal & string, Precision>, _FloorDecimal<AlignedN2Decimal & string, Precision>> extends [infer NewDecimal, infer Underflow] ?
	_UpdateWholeResultWithSubtractDecimalResult<Subtract<N1Whole & string, N2Whole & string>, Underflow & boolean> extends infer NewWhole ? Str.AppendTwo<NewWhole & string, '.', NewDecimal & string> : never
	: never : never

/**
 * Infers and returns the length of given Tuple T.
 */
export type TupleLength<T extends unknown[]> = T extends {
	length: infer L;
} ? L : never;

/**
 * A table from StringDigits to an array that repeats the elements of given
 * array T digit amount of times.
 * 'deca' is used to represent an exponent of 10
 */
export type DigitToTup<T extends unknown[] = [unknown]> = {
	'0': [];
	'1': [...T];
	'2': [...T, ...T];
	'3': [...T, ...T, ...T];
	'4': [...T, ...T, ...T, ...T];
	'5': [...T, ...T, ...T, ...T, ...T];
	'6': [...T, ...T, ...T, ...T, ...T, ...T];
	'7': [...T, ...T, ...T, ...T, ...T, ...T, ...T];
	'8': [...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T];
	'9': [...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T];
	deca: [...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T, ...T];
};

/**
 * Builds a tuple of the length of a given string-number. The length can be used
 * to convert a string-number to a number.
 */
export type BuildTuple<N extends string, Deca extends unknown[] = [unknown]> = N extends Str.Append<infer First, Digit> ? First extends '' ? DigitToTup<Deca>[Digit & N] : N extends Str.Append<First, infer D> ? [...BuildTuple<First, DigitToTup<Deca>['deca']>, ...BuildTuple<D, Deca>] : never : never;

/**
 * Converts the given string-number to its corresponding number.
 * @note due to compiler limitations the maximum value is '9999'
 * @example
 * ToNumber<'13'> => 13
 * ToNumber<'5234'> => 5234
 */
export type ToNumber<N extends string> = TupleLength<BuildTuple<N>>;

/**
 * Returns the result of multiplying the given string number with the given digit.
 */
export type MultDigit<N1 extends string, D extends Digit> = N1 extends '0' ? '0' : N1 extends '1' ? D : D extends '0' ? '0' : D extends '1' ? N1 : D extends '2' ? Add<N1, N1> : D extends '3' ? Add<N1, Add<N1, N1>> : D extends '4' ? Add<N1, N1> extends infer T ? Add<string & T, string & T> : never : D extends '5' ? Add<N1, MultDigit<N1, '4'>> : D extends '6' ? MultDigit<N1, '3'> extends infer T ? Add<string & T, string & T> : never : D extends '7' ? Add<N1, MultDigit<N1, '6'>> : D extends '8' ? MultDigit<N1, '4'> extends infer T ? Add<string & T, string & T> : never : D extends '9' ? Add<N1, MultDigit<N1, '8'>> : never;

export type Mult<N1 extends string, N2 extends string> = N2 extends Str.Append<infer N2Start, Digit> ? N2Start extends '' ? MultDigit<N1, Digit & N2> : N2 extends Str.Append<N2Start, infer N2Last> ? Add<MultDigit<N1, Digit & N2Last>, Str.Append<Mult<N1, N2Start>, '0'>> : never : never;

export type AmountTimesIn<Small extends string, Large extends string, Am extends string = '0'> = Small extends Large ? [Add<Am, '1'>, '0'] : Subtract<Large, Small> extends never ? [Am, Large] : Subtract<Large, Small> extends infer Result ? AmountTimesIn<Small, string & Result, Add<Am, '1'>> : never;

export type AppendDigit<N extends string, D extends Digit> = N extends '0' ? D : Str.Append<N, D>;

export type Divide<N extends string, M extends string, Q extends string = '', R extends string = ''> = M extends '0' ? never : N extends Str.Append<infer Dig, infer Rest> ? AppendDigit<R, Digit & Dig> extends infer D ? AmountTimesIn<M, string & D> extends [infer B, infer NewR] ? Divide<Rest, M, AppendDigit<Q, Digit & B>, string & NewR> : never : never : [Q, R];
