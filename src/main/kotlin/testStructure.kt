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

fun main(args: Array<String>) {
    var batch = MNIST.buildBatch(1000).asSequence().filter { it.index == 0 }
    val nw = CNetwork().load("nets/nw05.net")!!
    var counter = 0
    batch.forEach {
        var r = nw.activate(it, 15.0)[0]
        r = round(r)
        if (r == 1.0) counter++
        println("${it.index} -> $r ")
    }
    val c1 = counter*1.0/batch.count()
    batch = MNIST.buildBatch(1000).asSequence().filter { it.index != 0 }
    counter = 0
    batch.forEach {
        var r = nw.activate(it, 15.0)[0]
        r = round(r)
        if (r == 0.0) counter++
//        println("${it.index} -> $r ")
    }
    val c2 = counter*1.0/batch.count()
    println("$c1, $c2")
}