import java.io.File
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

fun test(number: Int, nw: Network): Boolean {
    var batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index == number }
    var counter = 0
    batch.forEach {
        val r = nw.activate(it, 15.0)[0]
        if (r >= 0.5) counter++
//        println("${it.index} -> $r ")
    }
    val c1 = (counter*1.0/batch.count()*100).toInt()/100.0
    var c3 = counter.toDouble()
    batch = MNIST.buildBatch(1000, MNIST.mnistTestPath).asSequence().filter { it.index != number }
    counter = 0
    batch.forEach {
        var r = nw.activate(it, 15.0)[0]
        r = round(r)
        if (r < 0.5) counter++
//        println("${it.index} -> $r ")
    }
    val c2 = (counter*1.0/batch.count()*100).toInt()/100.0
    c3 += counter
    c3 = (c3/10).toInt()/100.0
    println("$c1, $c2, $c3")
    return !(c3 < 0.9 || c1 < 0.5 || c2 < 0.9)
}

fun main(args: Array<String>) {
    val number = 2
    var n = 1
    for (i in 1..100) {
        val path = "nets/nw$number$i.net"
        CNetwork().load(path)?.let {
            print("$i) ")
            val r = true//test(number, it)
            if (r) {
                if (n != i) {
                    saveAs(path, "nets/nw$number$n.net")
                    delete(path)
                }
                n++
            } else delete(path)
        }
    }
}

fun delete(name: String) {
    val f = File(name)
    f.delete()
}
