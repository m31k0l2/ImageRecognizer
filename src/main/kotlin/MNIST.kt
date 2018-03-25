import java.io.File
import java.util.*

object MNIST {
    private val mnistTrainPath = "/home/melkor/mnist_png/training"
    private val mnistTestPath = "/home/melkor/mnist_png/testing"
    private val dir = File(mnistTestPath)

    fun buildBatch(size: Int): List<Image> {
        return (0 until size/10).flatMap {
            dir.listFiles().map {
                it.listFiles().toList().shuffled().first()
            }.map { Image(it) }
        }
    }

    fun next(): Image {
        val files = File(mnistTestPath).listFiles().flatMap { it.listFiles().toList() }
        return Image(files[Random().nextInt(files.size)])
    }
}
