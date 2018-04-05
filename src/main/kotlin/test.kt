import java.io.File

fun main(args: Array<String>) {
    val batch = MNIST.buildBatch(1000)
    var err = 0
    val dir = File("nets")
    val nets = dir.listFiles().mapNotNull {
        NetworkIO().load(it.canonicalPath)
    }
    batch.forEach { image ->
        print("${image.index} -> ")
        val r = nets.map { it.activate(image).map { if (it > 0.5) it else 0.0 } }.reduce { acc, list -> acc.zip(list, { a, b -> a + b }) }
        val m = r.max()!!
        val i = r.indexOf(m)
        if (i != image.index) {
            err++
            println("$i [${(m*10).toInt()/10.0} / ${(r[image.index]*10).toInt()/10.0}]")
        } else {
            println("$i")
        }
    }
    println("Ошибочно распознано: $err / ${batch.size}")
    println("Успех: ${((batch.size-err)*1000.0/batch.size).toInt()/10.0}%")
}