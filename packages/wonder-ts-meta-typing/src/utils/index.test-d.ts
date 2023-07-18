import { expectType } from 'tsd';
import { Extends, NotExtends, Check, Validate, Pred, Not } from '.';
import { IsPositive } from '../number/index';

expectType<Extends<1, number>>(true)
expectType<Extends<1, string>>(false)
expectType<Extends<1, number, "a", "b">>("a")
expectType<Extends<1, string, "a", "b">>("b")


expectType<NotExtends<1, number>>(false)
expectType<NotExtends<1, string>>(true)
expectType<NotExtends<1, number, "a", "b">>("b")
expectType<NotExtends<1, string, "a", "b">>("a")

expectType<Check<IsPositive<-1>>>(this as never)
expectType<Check<IsPositive<1>>>(this as unknown)

expectType<Validate<IsPositive<1>>>(true)
expectType<Validate<IsPositive<-1>>>(this as never)

expectType<Pred<IsPositive<1>>>(true)
expectType<Pred<IsPositive<-1>>>(false)

expectType<Not<IsPositive<-1>>>(true)
expectType<Not<true>>(false)
expectType<Not<false>>(true)
expectType<Not<boolean>>(this as never)
