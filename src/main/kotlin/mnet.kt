import java.util.*

//fun main(args: Array<String>) {
//    val batch = MNIST.buildBatch(100)
//    var names = listOf("012", "3456", "789")
//    names = (0..2).flatMap { n -> names.map { "${it}_$n" } }.sorted()
//    val nets = names.map { "nets/nw$it.net" }.map { NetworkIO().load(it)!! }
//    var counter = 0
//    batch.forEach {
//        val results = nets.mapNotNull { nw ->
//            val o = nw.activate(it)
//            val x = o.max()!!
//            if (x > 0.9) {
//                o.indexOf(x)
//            } else {
//                null
//            }
//        }
//        val r = (0..9).map { n -> n to results.filter { it == n }.size }.maxBy { it.second }?.first
//        println("${it.index} -> $r")
//        counter += if (it.index == r) 1 else 0
//    }
//    println("$counter / ${batch.size}")
////    val nw = NetworkIO().load("nets/nw.net", false)!! as MNetwork
////    batch.forEach {
////        val o = nw.activate(it)
////        val x = o.max()!!
////        print("${it.index} -> ")
////        if (x > 0.5) {
////            println(o.indexOf(x))
////        } else {
////            println("?")
////        }
////        println(o)
////    }
//}

fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(100)
    val groups = listOf("012", "3456", "789").map {
        (0..5).map { n -> "${it}_$n" }.map { "nets/nw$it.net" }.map { NetworkIO().load(it)!! }
    }
    val image = batch[Random().nextInt(batch.size)]
    val r = groups.map { it.map { it.activate(image) } }
    val getGroupResult = { pos: Int, size: Int -> r[pos].map { it.subList(0, size) }.reduce { acc, list -> (0 until acc.size).map { acc[it] + list[it]} }.map { it/r[0].size }}
    val group1 = getGroupResult(0, 3)
    val group2 = getGroupResult(1, 4)
    val group3 = getGroupResult(2, 3)
    var res = listOf(group1, group2, group3).flatMap { it }
    res = res.map { (it/res.sum()*1000).toInt()/10.0 }
    println("${image.index} = ${res.indexOf(res.max())} ")
    res.forEachIndexed { i, d ->
        println("$i -> $d")
    }
//    val nets = names.map { "nets/nw$it.net" }.map { NetworkIO().load(it)!! }
//    var counter = 0
//    batch.forEach {
//        val results = nets.mapNotNull { nw ->
//            val o = nw.activate(it)
//            val x = o.max()!!
//            if (x > 0.5) {
//                o.indexOf(x)
//            } else {
//                null
//            }
//        }
//        val r = (0..9).map { n -> n to results.filter { it == n }.size }.maxBy { it.second }?.first
//        println("${it.index} -> $r")
//        counter += if (it.index == r) 1 else 0
//    }
//    println("$counter / ${batch.size}")
}