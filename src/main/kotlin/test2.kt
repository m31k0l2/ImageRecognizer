fun main(args: Array<String>) {
    val nameSet = (102..987).mapNotNull {
        val a = it % 10
        val b = it/10 % 10
        val c = it/100 % 10
        listOf(a, b, c).sorted().takeIf { a != b && a != c && b != c }
    }.toSet()
    val nets = (0..9).map { i ->
        nameSet.filter { it.indexOf(i) != -1 }.map { "nets/${it[0]}${it[1]}${it[2]}.net" }.mapNotNull { NetworkIO().load(it) }
    }
    val batch = MNIST.buildBatch(1000)
    var err = 0
    batch.forEach { image ->
        print("${image.index} -> ")
        val r = nets.mapIndexed { n, nw -> nw.map { it.activate(image) }.map { it[n] }.sorted()[nw.size/2-1] }
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