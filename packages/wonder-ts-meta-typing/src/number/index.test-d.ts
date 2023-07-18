import { expectType } from 'tsd';
import { Add, Subtract, Mult, Div, DivMod, IsPositive, IsNatural, IsNegative, Inc, Decr, Min, Max, Equal, NotEqual, GreaterThanOrEqual, GreaterThan, LessThan, LessThanOrEqual, InRange } from './';

expectType<IsPositive<-1>>(false)
expectType<IsPositive<1>>(true)

expectType<IsNatural<0>>(true)
expectType<IsNatural<-1>>(false)

expectType<IsNegative<-1>>(true)
expectType<IsNegative<0>>(false)
expectType<IsNegative<10>>(false)

expectType<Add<10, 8999>>(9009)
// expectType<Add<9999, 999>>()

expectType<Subtract<9999, 1>>(9998)
expectType<Subtract<10, 8999>>(this as never)


expectType<Mult<100, 70>>(7000)

expectType<Div<9, 3>>(3)
expectType<Div<9999, 3>>(3333)

expectType<DivMod<9999, 3>>([3333, 0])

expectType<Inc<9998>>(9999)

expectType<Decr<9998>>(9997)

expectType<Min<532, 9999>>(532)

expectType<Max<532, 99999>>(99999)
expectType<Max<532, 99999>>(99999)

expectType<Equal<532, 99999>>(false)
expectType<Equal<99999, 99999>>(true)

expectType<NotEqual<532, 99999>>(true)
expectType<NotEqual<99999, 99999>>(false)

expectType<GreaterThan<1, 99999>>(false)
expectType<GreaterThan<9999, 9999>>(false)
expectType<GreaterThan<9999, 9998>>(true)

expectType<GreaterThanOrEqual<9999, 9999>>(true)
expectType<GreaterThanOrEqual<9998, 9999>>(false)

expectType<LessThan<1, 99999>>(true)
expectType<LessThan<9999, 9999>>(false)
expectType<LessThan<9999, 9998>>(false)

expectType<LessThanOrEqual<9999, 9999>>(true)
expectType<LessThanOrEqual<9998, 9999>>(true)

expectType<InRange<4, 5, 10>>(false)
expectType<InRange<5, 5, 10>>(true)
expectType<InRange<6, 5, 10>>(true)
expectType<InRange<9999, 5, 9999>>(true)
