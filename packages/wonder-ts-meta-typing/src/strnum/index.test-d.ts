import { expectType } from 'tsd';
import { FromNumber, ToNumber, Add, Subtract, Mult, Divide, AddFloat, SubtractFloat, MaxFloat, MinFloat, LessThanFloat, LessThanOrEqualFloat, GreaterThanFloat, GreaterThanOrEqualFloat, InRangeFloat } from './index'

expectType<FromNumber<2>>("2")
expectType<FromNumber<1333>>("1333")
expectType<FromNumber<10000>>("10000")
expectType<FromNumber<0.10>>(this as never)

expectType<ToNumber<"9999">>(9999)
// expectType<ToNumber<"10000">>(this as never)

expectType<Add<"999", "9999">>("10998")

expectType<Subtract<"13", "8">>("5")
expectType<Subtract<"5371", "139">>("5232")
expectType<Subtract<"13000", "139">>("12861")
expectType<Subtract<"100", "101">>(this as never)

expectType<Mult<"100", "101">>("10100")

expectType<Divide<"100", "101">>(["0", "100"])
expectType<Divide<"300", "101">>(["2", "98"])

expectType<AddFloat<"1", "2">>("3")
expectType<AddFloat<"199.223", "10">>("209.223")
expectType<AddFloat<"10", "2.2">>("12.2")
expectType<AddFloat<"10.0", "2.0">>("12.0")
expectType<AddFloat<"10", "2.0">>("12.0")
expectType<AddFloat<"1.1", "2">>("3.1")
expectType<AddFloat<"1.1", "2.2">>("3.3")
// expectType<_RemoveZeroDecimal< AddFloat<"999999.1022", "2.2">>>("1000001.302")
expectType<AddFloat<"999999.1022", "2.2">>("1000001.302")
expectType<AddFloat<"999999.203000221000022", "2.223">>("1000001.426")

expectType<SubtractFloat<"10", "199.223">>(this as never)
expectType<SubtractFloat<"199.223", "10">>("189.223")
expectType<SubtractFloat<"199.223", "10.1">>("189.123")
expectType<SubtractFloat<"199.223", "10.57">>("188.653")
expectType<SubtractFloat<"999999.22322222", "10.57">>("999988.653")
expectType<SubtractFloat<"999999.22322222111111111", "10.57", 4>>("999988.6532")
expectType<SubtractFloat<"199", "10">>("189.0")
expectType<SubtractFloat<"a", "10">>(this as never)

expectType<MinFloat<"12", "2">>("2")
expectType<MinFloat<"2.2", "5">>("2.2")
expectType<MinFloat<"12.2", "2.5">>("2.5")
expectType<MinFloat<"99999.1111111111111222222222222222", "199999.1111111111111222222222222222">>("99999.1111111111111222222222222222")

expectType<MaxFloat<"12", "2">>("12")
expectType<MaxFloat<"2.2", "5">>("5")
expectType<MaxFloat<"12.2", "2.5">>("12.2")
expectType<MaxFloat<"99999.1111111111111222222222222222", "199999.1111111111111222222222222222">>("199999.1111111111111222222222222222")

expectType<GreaterThanFloat<"1", "9999999.2222222222222222">>(false)
expectType<GreaterThanFloat<"10.2", "9999">>(false)
expectType<GreaterThanFloat<"100", "99.1">>(true)
expectType<GreaterThanFloat<"100.1", "99.1">>(true)

expectType<GreaterThanOrEqualFloat<"9999", "9999">>(true)
expectType<GreaterThanOrEqualFloat<"9998.1", "9999.999">>(false)

expectType<LessThanFloat<"1", "9999999.2222222222222222">>(true)
expectType<LessThanFloat<"10.2", "9999">>(true)
expectType<LessThanFloat<"100", "99.1">>(false)
expectType<LessThanFloat<"100.1", "99.1">>(false)

expectType<LessThanOrEqualFloat<"9999", "9999">>(true)
expectType<LessThanOrEqualFloat<"9998.1", "9999.999">>(true)

expectType<InRangeFloat<"4", "5", "10">>(false)
expectType<InRangeFloat<"5.1", "5.1", "10.5">>(true)
expectType<InRangeFloat<"5.1", "5.2", "10">>(false)
expectType<InRangeFloat<"11.1", "5.2", "10">>(false)
expectType<InRangeFloat<"11.1", "5.2", "2">>(false)
expectType<InRangeFloat<"9999", "5", "9999.222222222222222">>(true)
expectType<InRangeFloat<"0.2", "0.0", "1.0">>(true)