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
        (0..2).map { n -> "${it}_$n" }.map { "nets/nw$it.net" }.map { NetworkIO().load(it)!! }
    }
    val image = batch[0]
    print("${image.index} -> ")
    val r = groups.map { it.map { it.activate(image) } }
    println(r)
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