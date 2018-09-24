import java.util.*
import kotlin.math.round

fun testMedianNet(number: Int, nw: Network, batch: Set<Image>, alpha: Double=15.0): Double {
    return batch.asSequence().groupBy { it.index }.map { (i, list) ->
        val r = list.asSequence().map { nw.activate(it, alpha)[0] }.average()
        if (i == number) r else 1 - r
    }.reduce { acc, d -> acc*d }
}

//fun main(args: Array<String>) {
//    while (true) {
//        print("Размер батча: ")
//        val size = Scanner(System.`in`).nextInt()
//        if (size < 30) return
//        val name = "nw0.net"
//        val nw = CNetwork().load("nets/$name")
//        for (number in 0..9) {
//            nw?.let {
//                val r = testMedianNet(number, it, MNIST.buildBatch(size))
//                println("$number -> $r")
//            }
//        }
//    }
//}

fun test(number: Int, nw: Network) {
    var batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index == number }
    var counter = 0
    batch.forEach {
        val r = nw.activate(it, 15.0)[0]
        if (r >= 0.5) counter++
//        println("${it.index} -> $r ")
    }
    val c1 = counter*1.0/batch.count()
    batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index != number }
    counter = 0
    batch.forEach {
        var r = nw.activate(it, 15.0)[0]
        r = round(r)
        if (r < 0.5) counter++
//        println("${it.index} -> $r ")
    }
    val c2 = counter*1.0/batch.count()
    println("$c1, $c2")
}

fun main(args: Array<String>) {
    val number = 2
    for (i in 1..10) {
        CNetwork().load("nets/nw$number$i.net")?.let {
            test(number, it)
        }
    }
}